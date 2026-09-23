// Particle positions/velocities stay in world units as the grid is refined.
export const GRID_SCALE = 2;
export const GRID_WIDTH = 160 * GRID_SCALE;
export const GRID_HEIGHT = 64 * GRID_SCALE;
export const PARTICLE_MASS = 1 / (GRID_SCALE * GRID_SCALE);
export const PARTICLE_CLEARANCE = 0.35 / GRID_SCALE;
export const FLUID_SHADER = /* wgsl */ `
const NX: u32 = ${GRID_WIDTH}u;
const NY: u32 = ${GRID_HEIGHT}u;
const R: f32 = ${GRID_SCALE.toFixed(1)};
const PARTICLE_MASS: f32 = ${PARTICLE_MASS.toFixed(8)};
const PARTICLE_CLEARANCE: f32 = ${PARTICLE_CLEARANCE.toFixed(8)};
const SCALE: f32 = 100000.0;
struct Params { dt:f32, time:f32, height:f32, period:f32, wind:f32, count:u32, gravity:f32, pad:f32 }
struct Particle { pos:vec2f, vel:vec2f, affine:vec4f, foam:f32, density:f32, pad:vec2f }
struct Cell { mass:atomic<i32>, mx:atomic<i32>, my:atomic<i32>, pad:atomic<i32> }
struct Pixel { mass:atomic<i32>, foam:atomic<i32> }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read_write> particles:array<Particle>;
@group(0) @binding(2) var<storage,read_write> cells:array<Cell>;
@group(0) @binding(3) var<storage,read_write> velocities:array<vec2f>;
@group(0) @binding(4) var<storage,read_write> pixels:array<Pixel>;
@group(0) @binding(5) var outputImage:texture_storage_2d<rgba8unorm,write>;
@group(0) @binding(6) var<storage,read_write> wet:array<f32>;
fn bed(x:f32)->f32 { return max(3.0,31.5-0.24*x); }
fn wall()->vec2f {
 let ramp=smoothstep(0.0,3.0,params.time);
 let phase=6.2831853*params.time/params.period;
 let stroke=params.height*2.25;
 // The piston stays outside the visible 128-cell viewport.
 return vec2f(157.0-stroke*(1.0-cos(phase))*ramp,-stroke*6.2831853/params.period*sin(phase)*ramp);
}
fn weights(f:vec2f)->array<vec2f,3> {
 return array<vec2f,3>(0.5*(1.5-f)*(1.5-f),0.75-(f-1.0)*(f-1.0),0.5*(f-0.5)*(f-0.5));
}
fn safeCell(p:vec2i)->u32 {return u32(clamp(p.y,0,i32(NY)-1))*NX+u32(clamp(p.x,0,i32(NX)-1));}
fn encode(v:f32)->i32 {return i32(clamp(v,-1000.0,1000.0)*SCALE);}
@compute @workgroup_size(64)
fn clearGrid(@builtin(global_invocation_id) id:vec3u) {
 let i=id.x;if(i>=NX*NY){return;}
 atomicStore(&cells[i].mass,0);atomicStore(&cells[i].mx,0);atomicStore(&cells[i].my,0);velocities[i]=vec2f(0);
}
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id:vec3u) {
 let i=id.x;if(i>=params.count){return;}let p=particles[i];
 let gridPos=p.pos*R;let base=vec2i(floor(gridPos-0.5));let w=weights(gridPos-vec2f(base));
 for(var y=0;y<3;y++){for(var x=0;x<3;x++){
  let cell=base+vec2i(x,y);let d=vec2f(cell)/R-p.pos;let weight=w[x].x*w[y].y;
  let affine=vec2f(dot(p.affine.xy,d),dot(p.affine.zw,d));
  let momentum=PARTICLE_MASS*weight*(p.vel+affine);let index=safeCell(cell);
  atomicAdd(&cells[index].mass,encode(PARTICLE_MASS*weight));
  atomicAdd(&cells[index].mx,encode(momentum.x));atomicAdd(&cells[index].my,encode(momentum.y));
 }}
}
@compute @workgroup_size(64)
fn pressure(@builtin(global_invocation_id) id:vec3u) {
 let i=id.x;if(i>=params.count){return;}let p=particles[i];
 let gridPos=p.pos*R;let base=vec2i(floor(gridPos-0.5));let w=weights(gridPos-vec2f(base));var density=0.0;
 for(var y=0;y<3;y++){for(var x=0;x<3;x++){
  density+=w[x].x*w[y].y*f32(atomicLoad(&cells[safeCell(base+vec2i(x,y))].mass))*R*R/SCALE;
 }}
 particles[i].density=density;
 let ratio=max(density/4.0,0.01);
 let pressureValue=clamp(650.0*(pow(ratio,7.0)-1.0),0.0,4000.0);
 let stress=4.0*R*R*params.dt*PARTICLE_MASS*pressureValue/max(density,0.1);
 for(var y=0;y<3;y++){for(var x=0;x<3;x++){
  let cell=base+vec2i(x,y);let d=vec2f(cell)/R-p.pos;let weight=w[x].x*w[y].y;
  let push=weight*stress*d;let index=safeCell(cell);
  atomicAdd(&cells[index].mx,encode(push.x));atomicAdd(&cells[index].my,encode(push.y));
 }}
}
@compute @workgroup_size(64)
fn updateGrid(@builtin(global_invocation_id) id:vec3u) {
 let i=id.x;if(i>=NX*NY){return;}
 let mass=f32(atomicLoad(&cells[i].mass))/SCALE;if(mass<0.0001){return;}
 var v=vec2f(f32(atomicLoad(&cells[i].mx)),f32(atomicLoad(&cells[i].my)))/(SCALE*mass);
 v.y-=params.gravity*params.dt;
 let x=f32(i%NX)/R;let y=f32(i/NX)/R;
 // Slip boundary on the sloping seabed; gravity drives the returning water.
 if(y<bed(x)+1.0/R){let n=normalize(vec2f(select(0.24,0.0,x>118.75),1));v-=n*min(dot(v,n),0.0);v*=pow(0.995,params.dt*240.0);}
 let piston=wall();if(x>piston.x-1.5){v.x=min(v.x,piston.y);}
 if(x<2.0){v.x=max(v.x,0.0);}if(y>61.0){v.y=min(v.y,0.0);}
 if(y>22.0){v.x-=params.wind*0.6*params.dt;}
 // A CFL safety limit avoids explosions after extreme live parameter changes.
 let speed=length(v);if(speed>70.0){v*=70.0/speed;}
 velocities[i]=v;
}
@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) id:vec3u) {
 let i=id.x;if(i>=params.count){return;}var p=particles[i];
 let gridPos=p.pos*R;let base=vec2i(floor(gridPos-0.5));let w=weights(gridPos-vec2f(base));var v=vec2f(0);var c=vec4f(0);
 for(var y=0;y<3;y++){for(var x=0;x<3;x++){
  let cell=base+vec2i(x,y);let d=vec2f(cell)/R-p.pos;let weighted=w[x].x*w[y].y*velocities[safeCell(cell)];
  v+=weighted;c+=vec4f(weighted.x*d,weighted.y*d);
 }}
 p.affine=clamp(c*(4.0*R*R*pow(0.99,params.dt*240.0)),vec4f(-40),vec4f(40));p.vel=v;p.pos+=v*params.dt;
 let piston=wall();
 if(p.pos.x>piston.x-0.5){p.pos.x=piston.x-0.5;p.vel.x=min(p.vel.x,piston.y);}
 p.pos=clamp(p.pos,vec2f(1.5,1.5),vec2f(157.5,61.5));
 let floorHeight=bed(p.pos.x)+PARTICLE_CLEARANCE;
 if(p.pos.y<floorHeight){p.pos.y=floorHeight;let n=normalize(vec2f(select(0.24,0.0,p.pos.x>118.75),1));p.vel-=n*min(dot(p.vel,n),0.0);}
 let strain=abs(p.affine.y+p.affine.z)+abs(p.affine.x-p.affine.w);
 let surface=1.0-smoothstep(2.6,3.9,p.density);
 let source=surface*smoothstep(2.0,8.0,length(v))*smoothstep(1.0,7.0,strain);
 p.foam=max(p.foam*exp(-params.dt*0.55),source);
 particles[i]=p;
}
@compute @workgroup_size(64)
fn clearPixels(@builtin(global_invocation_id) id:vec3u) {
 let i=id.x;if(i>=512u*256u){return;}atomicStore(&pixels[i].mass,0);atomicStore(&pixels[i].foam,0);
}
@compute @workgroup_size(64)
fn splat(@builtin(global_invocation_id) id:vec3u) {
 let i=id.x;if(i>=params.count){return;}let p=particles[i];let center=vec2f(p.pos.x*4.0,256.0-p.pos.y*4.0);
 let base=vec2i(floor(center));
 for(var dy=-3;dy<=3;dy++){for(var dx=-3;dx<=3;dx++){
  let pixel=base+vec2i(dx,dy);if(pixel.x<0||pixel.x>=512||pixel.y<0||pixel.y>=256){continue;}
  let d=vec2f(pixel)+0.5-center;let a=max(0.0,1.0-dot(d,d)/7.84);let amount=a*a*512.0*PARTICLE_MASS;
  let index=u32(pixel.y)*512u+u32(pixel.x);
  atomicAdd(&pixels[index].mass,i32(amount));atomicAdd(&pixels[index].foam,i32(amount*p.foam));
 }}
}
fn densityAt(p:vec2i)->f32 {
 let q=clamp(p,vec2i(0),vec2i(511,255));return f32(atomicLoad(&pixels[u32(q.y)*512u+u32(q.x)].mass))/512.0;
}
@compute @workgroup_size(64)
fn wetSand(@builtin(global_invocation_id) id:vec3u) {
 let x=id.x;if(x>=512u){return;}let y=i32(256.0-bed(f32(x)*0.25)*4.0)-2;
 wet[x]=max(wet[x]*exp(-params.dt*0.025),smoothstep(0.2,0.8,densityAt(vec2i(i32(x),y))));
}
fn hash(p:vec2u)->f32 {var h=p.x*1973u+p.y*9277u+89173u;h=(h^(h>>13u))*1274126177u;return f32(h&65535u)/65535.0;}
@compute @workgroup_size(8,8)
fn compose(@builtin(global_invocation_id) id:vec3u) {
 if(id.x>=512u||id.y>=256u){return;}let p=vec2i(id.xy);let point=vec2f(id.xy)+0.5;
 let sandY=256.0-bed(point.x*0.25)*4.0;let noise=hash(id.xy)-0.5;
 var color=mix(vec3f(0.64,0.79,0.84),vec3f(0.90,0.93,0.87),clamp(point.y/175.0,0.0,1.0));
 color+=noise*0.008;
 if(point.y>=sandY){
  let deep=point.y-sandY;let damp=wet[id.x]*exp(-deep/13.0);
  color=vec3f(0.87,0.79,0.61)-damp*vec3f(0.17,0.15,0.10)+noise*0.046;
  color+=sin(deep*0.14+sin(point.x/66.0))*0.008;
 }else{
  let density=densityAt(p);
  if(density>0.40){
   let dx=densityAt(p+vec2i(1,0))-densityAt(p-vec2i(1,0));
   let dy=densityAt(p+vec2i(0,1))-densityAt(p-vec2i(0,1));
   let deep=clamp((point.y-150.0)/100.0,0.0,1.0);
   var water=mix(vec3f(0.09,0.56,0.57),vec3f(0.025,0.25,0.34),deep);
   let surface=1.0-smoothstep(0.45,1.15,density);
   water+=vec3f(0.07,0.16,0.12)*surface+clamp(dy-dx*0.4,-0.4,0.8)*0.09*surface;
   let index=id.y*512u+id.x;
   let foam=f32(atomicLoad(&pixels[index].foam))/(512.0*max(density,0.01));
   water=mix(water,vec3f(0.91,0.97,0.90),clamp(foam*(0.78+noise*0.35)*(0.15+0.85*surface),0.0,0.95));
   color=mix(color,water,smoothstep(0.40,0.68,density));
  }
 }
 // Storage texture contains sRGB values; the Three.js material decodes these.
 textureStore(outputImage,p,vec4f(clamp(color,vec3f(0),vec3f(1)),1));
}
`;
