import * as THREE from "three/webgpu";
import { texture, vec2, uv, mix, step } from "three/tsl";
import { FluidCore, WIDTH, HEIGHT } from "./fluid-core";
import type { OceanSettings } from "./ocean";

export class GpuOcean {
  readonly renderer: THREE.WebGPURenderer;
  private image = new THREE.StorageTexture(WIDTH, HEIGHT);
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 2);
  private material = new THREE.MeshBasicNodeMaterial();
  private geometry = new THREE.PlaneGeometry(2, 2);
  private core!: FluidCore;
  private disposed = false;
  private failed = false;
  get time() { return this.core?.time ?? 0; }
  get count() { return this.core.count; }
  private constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGPURenderer({ canvas, antialias: false, alpha: false, requiredLimits: { maxStorageBuffersPerShaderStage: 5 } });
    this.renderer.setPixelRatio(1); this.renderer.setSize(WIDTH, HEIGHT, false);
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.image.minFilter = THREE.NearestFilter; this.image.magFilter = THREE.NearestFilter;
    this.image.generateMipmaps = false;
    const encoded = texture(this.image, vec2(uv().x, uv().y.oneMinus())).rgb;
    this.material.colorNode = mix(encoded.div(12.92), encoded.add(0.055).div(1.055).pow(2.4), step(0.04045, encoded));
    this.material.toneMapped = false; this.material.depthTest = false; this.material.depthWrite = false;
    this.scene.add(new THREE.Mesh(this.geometry, this.material)); this.camera.position.z = 1;
  }
  static async create(canvas: HTMLCanvasElement, onLost: (message: string) => void) {
    if (!navigator.gpu) throw new Error("このブラウザではWebGPUが使えません。対応ブラウザで開くか、簡易表示に切り替えてください。");
    const ocean = new GpuOcean(canvas);
    try {
      await ocean.renderer.init();
      // A single pinned Three.js backend boundary exposes its native resources.
      const backend = ocean.renderer.backend as unknown as {
        isWebGPUBackend: boolean; device: GPUDevice; get(texture: THREE.Texture): { texture: GPUTexture };
      };
      if (!backend.isWebGPUBackend) throw new Error("WebGPUを初期化できませんでした。簡易表示に切り替えられます。");
      ocean.renderer.initTexture(ocean.image);
      ocean.core = await FluidCore.create(backend.device, backend.get(ocean.image).texture);
      const fail = (message: string) => { if (!ocean.disposed && !ocean.failed) { ocean.failed = true; onLost(message); } };
      void backend.device.lost.then(() => fail("GPUとの接続が切れました。再試行するか、簡易表示に切り替えてください。"));
      backend.device.addEventListener("uncapturederror", (event: GPUUncapturedErrorEvent) => {
        console.error("Fluid GPU error", event.error); fail("GPUの計算でエラーが発生しました。再試行するか、簡易表示に切り替えてください。");
      });
      await ocean.renderer.compileAsync(ocean.scene, ocean.camera);
      return ocean;
    } catch (error) { ocean.dispose(); throw error; }
  }
  render(dt: number, settings: OceanSettings) {
    if (this.disposed || this.failed) return;
    this.core.frame(dt, settings); this.renderer.render(this.scene, this.camera);
  }
  reset() { this.core.reset(); }
  snapshot() {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL("image/png");
  }
  dispose() {
    if (this.disposed) return; this.disposed = true;
    this.core?.dispose(); this.image.dispose(); this.material.dispose(); this.geometry.dispose(); this.renderer.dispose();
  }
}
