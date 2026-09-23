import { FLUID_SHADER, GRID_SCALE, GRID_WIDTH, GRID_HEIGHT } from "./fluid-shaders";
import type { OceanSettings } from "./ocean";
export const WIDTH = 512, HEIGHT = 256, GRID_CELLS = GRID_WIDTH * GRID_HEIGHT;
export const FIXED_DT = 1 / (240 * GRID_SCALE);
const MAX_STEPS = 24 * GRID_SCALE;
const SLOTS = MAX_STEPS + 2;
export function initialParticles() {
  const values: number[] = [];
  for (let x = 2; x < 156.5; x += 0.5) {
    const bottom = Math.max(3, 31.5 - x * 0.24);
    for (let y = bottom + 0.55; y < 23.75; y += 0.5) {
      // Subdivide each original particle, preserving its mass and center.
      for (let sx = 0; sx < GRID_SCALE; sx++) for (let sy = 0; sy < GRID_SCALE; sy++) {
        const dx = ((sx + 0.5) / GRID_SCALE - 0.5) * 0.5;
        const dy = ((sy + 0.5) / GRID_SCALE - 0.5) * 0.5;
        values.push(x + dx, y + dy, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0);
      }
    }
  }
  return new Float32Array(values);
}
/** Native compute shares the Three.js device/StorageTexture, with no frame readback. */
export class FluidCore {
  time = 0;
  readonly count: number;
  readonly particleBuffer: GPUBuffer;
  readonly buffers: GPUBuffer[];
  private pipelines = new Map<string, GPUComputePipeline>();
  private bindGroup!: GPUBindGroup;
  private uniform: GPUBuffer;
  private uniforms = new ArrayBuffer(SLOTS * 256);
  private initial: Float32Array;
  private accumulator = 0;
  private wetBuffer: GPUBuffer;
  private disposed = false;
  private constructor(readonly device: GPUDevice, readonly output: GPUTexture) {
    this.initial = initialParticles(); this.count = this.initial.length / 12;
    const buffer = (label: string, size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) => device.createBuffer({ label, size, usage });
    this.particleBuffer = buffer("water particles", this.initial.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.uniform = buffer("substep parameters", this.uniforms.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const cells = buffer("atomic grid", GRID_CELLS * 16);
    const velocities = buffer("grid velocities", GRID_CELLS * 8);
    const pixels = buffer("particle raster", WIDTH * HEIGHT * 16);
    const optics = buffer("top-light optical paths", WIDTH * HEIGHT * 5 * 4);
    this.wetBuffer = buffer("wet beach", WIDTH * 4);
    this.buffers = [this.particleBuffer, this.uniform, cells, velocities, pixels, this.wetBuffer, optics];
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 32 } },
      ...[1,2,3,4,6,7].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
    ] });
    this.bindGroup = device.createBindGroup({ layout, entries: [
      { binding: 0, resource: { buffer: this.uniform, size: 32 } },
      { binding: 1, resource: { buffer: this.particleBuffer } },
      { binding: 2, resource: { buffer: cells } },
      { binding: 3, resource: { buffer: velocities } },
      { binding: 4, resource: { buffer: pixels } },
      { binding: 5, resource: output.createView() },
      { binding: 6, resource: { buffer: this.wetBuffer } },
      { binding: 7, resource: { buffer: optics } },
    ] });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.reset();
  }
  private pipelineLayout: GPUPipelineLayout;
  static async create(device: GPUDevice, output: GPUTexture) {
    const core = new FluidCore(device, output);
    try {
      const module = device.createShaderModule({ label: "2D coastal fluid", code: FLUID_SHADER });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter(m => m.type === "error");
      if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
      const names = ["clearGrid", "scatter", "pressure", "updateGrid", "gather", "clearPixels", "splat", "wetSand", "opticalDepth", "compose"];
      await Promise.all(names.map(async name => {
        core.pipelines.set(name, await device.createComputePipelineAsync({ label: name, layout: core.pipelineLayout, compute: { module, entryPoint: name } }));
      }));
      return core;
    } catch (error) { core.dispose(); throw error; }
  }
  reset() {
    if (this.disposed) return;
    this.time = 0; this.accumulator = 0;
    this.device.queue.writeBuffer(this.particleBuffer, 0, this.initial as Float32Array<ArrayBuffer>);
    const command = this.device.createCommandEncoder(); command.clearBuffer(this.wetBuffer);
    this.device.queue.submit([command.finish()]);
  }
  frame(dt: number, settings: OceanSettings) {
    if (this.disposed) return;
    this.accumulator = Math.min(0.1, this.accumulator + Math.min(Math.max(dt, 0), 0.05) * settings.speed);
    const steps = Math.min(MAX_STEPS, Math.floor((this.accumulator + 1e-8) / FIXED_DT));
    this.accumulator -= steps * FIXED_DT;
    const encoder = this.device.createCommandEncoder({ label: "fluid frame" });
    const values = new DataView(this.uniforms);
    const params = (slot: number, step: number, optics = 0) => {
      const offset = slot * 256;
      [step, this.time, settings.height, settings.period, settings.wind].forEach((v, i) => values.setFloat32(offset + i * 4, v, true));
      values.setUint32(offset + 20, this.count, true); values.setFloat32(offset + 24, 18, true); values.setUint32(offset + 28, optics, true);
    };
    const dispatch = (name: string, count: number, slot: number, y = 1) => {
      const pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.get(name)!);
      pass.setBindGroup(0, this.bindGroup, [slot * 256]); pass.dispatchWorkgroups(count, y); pass.end();
    };
    for (let i = 0; i < steps; i++) {
      this.time += FIXED_DT; params(i, FIXED_DT);
      dispatch("clearGrid", Math.ceil(GRID_CELLS / 64), i);
      dispatch("scatter", Math.ceil(this.count / 64), i);
      dispatch("pressure", Math.ceil(this.count / 64), i);
      dispatch("updateGrid", Math.ceil(GRID_CELLS / 64), i);
      dispatch("gather", Math.ceil(this.count / 64), i);
    }
    params(steps, steps * FIXED_DT);
    dispatch("clearPixels", WIDTH * HEIGHT / 64, steps);
    dispatch("splat", Math.ceil(this.count / 64), steps);
    dispatch("wetSand", WIDTH / 64, steps);
    dispatch("opticalDepth", 640 / 64, steps);
    dispatch("compose", WIDTH / 8, steps, HEIGHT / 8);
    this.device.queue.writeBuffer(this.uniform, 0, this.uniforms, 0, (steps + 1) * 256);
    this.device.queue.submit([encoder.finish()]);
  }
  dispose() {
    if (this.disposed) return; this.disposed = true;
    for (const buffer of this.buffers) buffer.destroy();
  }
}
