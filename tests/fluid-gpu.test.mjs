import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import ts from 'typescript';
import { create, globals } from 'webgpu';

async function loadCore() {
  const dir = await mkdtemp(join(tmpdir(), 'shore-fluid-test-'));
  for (const name of ['fluid-shaders', 'fluid-core', 'gpu-ocean']) {
    const source = await readFile(new URL(`../lib/${name}.ts`, import.meta.url), 'utf8');
    const js = ts.transpile(source, { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 }).replace('"./fluid-shaders"', '"./fluid-shaders.mjs"').replace('"./fluid-core"', '"./fluid-core.mjs"').replace('"three/webgpu"', JSON.stringify(pathToFileURL(resolve('node_modules/three/build/three.webgpu.js')).href)).replace('"three/tsl"', JSON.stringify(pathToFileURL(resolve('node_modules/three/build/three.tsl.js')).href));
    await writeFile(join(dir, `${name}.mjs`), js);
  }
  const core = await import(pathToFileURL(join(dir, 'fluid-core.mjs')).href);
  const constants = await import(pathToFileURL(join(dir, 'fluid-shaders.mjs')).href);
  const view = await import(pathToFileURL(join(dir, 'gpu-ocean.mjs')).href);
  await rm(dir, { recursive: true }); return { ...core, ...view, ...constants };
}

test('GPU fluid: finite particles, beach collisions, waves, reset and pixel output', { timeout: 180000 }, async t => {
  Object.assign(globalThis, globals);
  const gpu = create(process.platform === 'darwin' ? ['backend=metal'] : []);
  // Dawn's native instance must remain reachable until all GPU work completes.
  Object.defineProperty(globalThis.navigator, 'gpu', { value: gpu, configurable: true });
  const adapter = await gpu.requestAdapter();
  assert(adapter, 'A real GPU adapter is required; run this test with GPU access');
  const device = await adapter.requestDevice();
  const errors = []; device.addEventListener('uncapturederror', e => errors.push(e.error.message));
  const { FluidCore, initialParticles, PARTICLE_MASS, PARTICLE_CLEARANCE, GRID_WIDTH, GRID_HEIGHT } = await loadCore();
  const output = device.createTexture({ size: [512,256], format:'rgba8unorm', usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.TEXTURE_BINDING });
  let fluid;
  try {
    fluid = await FluidCore.create(device, output);
    const settings={height:2.2,period:6,wind:.3,speed:1};
    const readParticles=async()=>{
      const buf=device.createBuffer({size:fluid.particleBuffer.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      const enc=device.createCommandEncoder();enc.copyBufferToBuffer(fluid.particleBuffer,0,buf,0,buf.size);device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);const data=new Float32Array(buf.getMappedRange().slice(0));buf.unmap();buf.destroy();return data;
    };
    const initial=initialParticles();
    assert.equal(GRID_WIDTH,320);assert.equal(GRID_HEIGHT,128);
    assert.equal(fluid.count,6557*4);
    assert.equal(fluid.count*PARTICLE_MASS,6557,'Refinement must preserve total water mass');
    fluid.frame(0,settings);assert.deepEqual(await readParticles(),initial);
    const start=performance.now();
    let maxSpeed=0,highest=0,coastalMotion=0;
    for(let frame=0;frame<360;frame++){
      fluid.frame(1/30,settings);
      // Limit queue depth, matching a browser rather than enqueueing 12 s at once.
      if(frame%3===0)await device.queue.onSubmittedWorkDone();
      if(frame%60===59){
        const data=await readParticles();
        for(let i=0;i<data.length;i+=12){
          const x=data[i],y=data[i+1];
          assert(data.slice(i,i+12).every(Number.isFinite),'NaN or infinite fluid state');
          assert(x>=1.49&&x<=157.51&&y<=61.51,'Particle escaped domain');
          assert(y>=Math.max(3,31.5-.24*x)+PARTICLE_CLEARANCE-.01,'Particle penetrated sand');
          maxSpeed=Math.max(maxSpeed,Math.hypot(data[i+2],data[i+3]));
          if(x>40&&x<95){highest=Math.max(highest,y);coastalMotion+=Math.abs(data[i+2]);}
        }
      }
    }
    assert(Math.abs(fluid.time-12)<1e-6);assert(maxSpeed>1&&maxSpeed<75);
    assert(highest>24,'Waves must rise above initial sea level');assert(coastalMotion>100,'Waves must reach the coast');
    const readPixels=async()=>{
      const pixels=device.createBuffer({size:512*256*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      const enc=device.createCommandEncoder();enc.copyTextureToBuffer({texture:output},{buffer:pixels,bytesPerRow:2048},{width:512,height:256});device.queue.submit([enc.finish()]);
      await pixels.mapAsync(GPUMapMode.READ);const rgba=new Uint8Array(pixels.getMappedRange().slice(0));pixels.unmap();pixels.destroy();return rgba;
    };
    const rgba=await readPixels();
    assert.equal(rgba.length,512*256*4);assert.equal(rgba[3],255);assert.notEqual(rgba[0],rgba[4*(230*512)]);
    if(process.env.SHORE_TEST_IMAGE){
      const rgb=Buffer.alloc(512*256*3);for(let i=0;i<512*256;i++)rgb.set(rgba.subarray(i*4,i*4+3),i*3);
      await writeFile(process.env.SHORE_TEST_IMAGE,Buffer.concat([Buffer.from('P6\n512 256\n255\n'),rgb]));
    }
    t.diagnostic(`${fluid.count} particles, 12 seconds of fluid in ${((performance.now()-start)/1000).toFixed(2)} seconds; peak speed ${maxSpeed.toFixed(2)}, crest y ${highest.toFixed(2)}`);
    fluid.reset();fluid.frame(0,settings);assert.equal(fluid.time,0);assert.deepEqual(await readParticles(),initial);
    for(const height of [.2,3])for(const period of [3,10]){
      for(let i=0;i<15;i++){fluid.frame(1/30,{...settings,height,period,speed:2});await device.queue.onSubmittedWorkDone();}
      await device.queue.onSubmittedWorkDone();assert((await readParticles()).every(Number.isFinite));
    }
    // An isolated airborne water particle must survive the surface threshold
    // as a white glint. Equally fast dense water must retain its water color.
    const fixture=initial.slice();
    fixture.set([80.125,40.125,5,-2,0,0,0,0,0.9,0.5,0,0],0);
    fixture.set([90.125,40.125,5,-2,0,0,0,0,0,4,0,0],12);
    device.queue.writeBuffer(fluid.particleBuffer,0,fixture);
    fluid.frame(0,settings);const drops=await readPixels();
    const pixel=(x,y)=>Array.from(drops.slice((y*512+x)*4,(y*512+x)*4+3));
    const white=pixel(320,95),dense=pixel(360,95);
    assert(Math.min(...white)>240,'Airborne droplet should be white');
    assert(Math.max(...white)-Math.min(...white)<12,'Spray should be neutral, not cyan');
    assert(dense[0]<230,'Speed alone must not whiten dense water');
    // A submerged air pocket must not reset top-light attenuation. Its removal
    // of a little water can transmit slightly more light, but cannot light a column.
    const pool=initial.slice();
    for(let i=0;i<fluid.count;i++)pool.set([70+(i%224)*.25,15+Math.floor(i/224)*.25,0,0,0,0,0,0,0,4,0,0],i*12);
    device.queue.writeBuffer(fluid.particleBuffer,0,pool);fluid.frame(0,settings);
    const solid=await readPixels();
    for(let i=0;i<fluid.count;i++)if(Math.hypot(pool[i*12]-95,pool[i*12+1]-34)<1.5){pool[i*12]=145;pool[i*12+1]=35;}
    device.queue.writeBuffer(fluid.particleBuffer,0,pool);fluid.frame(0,settings);
    const pocket=await readPixels();
    const below=(184*512+380)*4;
    for(let channel=0;channel<3;channel++)assert(Math.abs(solid[below+channel]-pocket[below+channel])<=12,'Air pocket reset accumulated light attenuation');
    const hole=(120*512+380)*4;
    assert(pocket[hole]>solid[hole]+20,'Air pocket must reveal the unchanged sky background');
    // Add a sheet of water above an otherwise unchanged sample. It must shade
    // the water below, even with an air gap between sheet and sea.
    const overhead=pool.slice();
    for(let i=0;i<320;i++)overhead.set([90+(i%40)*.25,48+Math.floor(i/40)*.25,0,0,0,0,0,0,0,4,0,0],i*12);
    device.queue.writeBuffer(fluid.particleBuffer,0,overhead);fluid.frame(0,settings);
    const shaded=await readPixels();
    const brightness=image=>image[below]+image[below+1]+image[below+2];
    assert(brightness(shaded)<brightness(pocket)-2,'Overhead water must cast a shadow across the air gap');
    // With all water outside the viewport, this is the reference sky. The air
    // pocket must match it even when an extra water sheet is placed above it.
    const empty=initial.slice();
    for(let i=0;i<fluid.count;i++){empty[i*12]=145;empty[i*12+1]=35;}
    device.queue.writeBuffer(fluid.particleBuffer,0,empty);fluid.frame(0,settings);
    const sky=await readPixels();
    assert.deepEqual(pocket.slice(hole,hole+3),sky.slice(hole,hole+3),'Air background was shadowed by water');
    assert.deepEqual(shaded.slice(hole,hole+3),sky.slice(hole,hole+3),'Overhead sheet darkened air instead of just water');
    assert.deepEqual(errors,[],'WebGPU validation errors');
  } finally {fluid?.dispose();output.destroy();device.destroy();delete globalThis.navigator.gpu;}
});


test('Three.js renders the shared GPU texture without readback', { timeout: 30000 }, async () => {
  Object.assign(globalThis, globals);
  const gpu = create(process.platform === 'darwin' ? ['backend=metal'] : []);
  Object.defineProperty(globalThis.navigator, 'gpu', { value: gpu, configurable: true });
  globalThis.requestAnimationFrame = callback => setTimeout(() => callback(performance.now()), 0);
  globalThis.self = { requestAnimationFrame: () => 0, cancelAnimationFrame: () => {} };
  const { GpuOcean } = await loadCore();
  let target, device;
  const canvas = { width:512, height:256, style:{}, addEventListener(){}, removeEventListener(){}, setAttribute(){}, getContext(){return context;} };
  const context = {
    canvas,
    configure(config) {
      device = config.device;
      target?.destroy();
      target = device.createTexture({ size:[512,256], format:config.format, usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC });
    },
    unconfigure(){}, getCurrentTexture(){return target;},
  };
  let ocean;
  const failures=[];
  try {
    ocean=await GpuOcean.create(canvas, message=>failures.push(message));
    ocean.render(1/30,{height:2.2,period:6,wind:.3,speed:1});
    await device.queue.onSubmittedWorkDone();
    const buf=device.createBuffer({size:512*256*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const command=device.createCommandEncoder();command.copyTextureToBuffer({texture:target},{buffer:buf,bytesPerRow:2048},{width:512,height:256});device.queue.submit([command.finish()]);
    await buf.mapAsync(GPUMapMode.READ);const data=new Uint8Array(buf.getMappedRange().slice(0));buf.unmap();buf.destroy();
    assert.equal(data[3],255);
    const sky=Array.from(data.slice(0,3));const sand=Array.from(data.slice(250*512*4,250*512*4+3));
    assert.notDeepEqual(sky,sand,'Canvas output must show the fluid scene');
    // Preferred canvas format is BGRA on Metal: sky has more blue than sand.
    const blue=target.format.startsWith('bgra')?0:2;
    assert(data[blue]>data[250*512*4+blue],'Sky must be above the beach, not flipped');
    assert.deepEqual(failures,[]);
  } finally {
    ocean?.dispose();target?.destroy();device?.destroy();delete globalThis.navigator.gpu;delete globalThis.self;delete globalThis.requestAnimationFrame;
  }
});
