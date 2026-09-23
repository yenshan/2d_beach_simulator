export type OceanSettings = { height: number; period: number; wind: number; speed: number };
type Spray = { x: number; y: number; vx: number; vy: number; life: number };
type Wave = { x: number; amplitude: number; breaking: number; collapse: number };
type Point = { x: number; y: number };
const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
const smooth = (v: number) => { const q = clamp(v); return q * q * (3 - 2 * q); };

/** Procedural plunging breakers with explicit overhanging sheets of water. */
export class Ocean {
  readonly width = 512;
  readonly height = 256;
  time = 0;
  private image: ImageData;
  private grain = new Float32Array(512 * 256);
  private bed = new Float32Array(512);
  private surface = new Float32Array(512);
  private wet = new Float32Array(512);
  private foam = new Float32Array(512);
  private nextFoam = new Float32Array(512);
  private lip = new Uint8Array(512 * 256);
  private spray: Spray[] = [];
  private emission = 0;
  private seed = 72843;
  private random() {
    this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0;
    return this.seed / 4294967296;
  }
  constructor(private ctx: CanvasRenderingContext2D) {
    this.image = ctx.createImageData(this.width, this.height);
    for (let i = 0; i < this.grain.length; i++) this.grain[i] = this.random();
    for (let x = 0; x < 512; x++) this.bed[x] = 130 + x * 0.24 + 2 * Math.sin(x / 79);
  }
  reset() {
    this.time = 0; this.foam.fill(0); this.nextFoam.fill(0); this.wet.fill(0);
    this.spray = []; this.emission = 0; this.seed = 72843;
  }
  private sheet(points: Point[]) {
    const ymin = Math.max(0, Math.floor(Math.min(...points.map(p => p.y))));
    const ymax = Math.min(255, Math.ceil(Math.max(...points.map(p => p.y))));
    for (let y = ymin; y <= ymax; y++) {
      const cuts: number[] = [];
      for (let j = 0; j < points.length; j++) {
        const a = points[j], b = points[(j + 1) % points.length];
        if ((a.y <= y + 0.5 && b.y > y + 0.5) || (b.y <= y + 0.5 && a.y > y + 0.5))
          cuts.push(a.x + (y + 0.5 - a.y) * (b.x - a.x) / (b.y - a.y));
      }
      cuts.sort((a,b) => a-b);
      for (let j = 0; j + 1 < cuts.length; j += 2)
        for (let x = Math.max(0, Math.ceil(cuts[j])); x < Math.min(512, cuts[j+1]); x++) this.lip[y * 512 + x] = 1;
    }
  }
  render(dt: number, settings: OceanSettings) {
    const step = clamp(dt, 0, 0.06) * settings.speed;
    this.time += step;
    const t = this.time, data = this.image.data, amplitude = settings.height * 18;
    const waves: Wave[] = [];
    // Travel time is integrated along the profile; breakers slow in the shallows.
    const travelTime = 13.2, offset = 5.5;
    const first = Math.floor((t + offset - travelTime) / settings.period);
    const last = Math.floor((t + offset) / settings.period);
    for (let n = first; n <= last; n++) {
      const age = t + offset - n * settings.period;
      if (age < 0 || age > travelTime) continue;
      const x = age < 7 ? 570 - age * 45 : 255 - (age - 7) * 30;
      const breaking = smooth((282 - x) / 88) * smooth((settings.height - 0.45) / 0.95);
      const collapse = smooth((204 - x) / 67);
      const set = 0.88 + 0.12 * Math.sin(n * 2.4 + 0.8);
      const shoal = 0.72 + 0.42 * Math.exp(-(((x - 258) / 95) ** 2));
      waves.push({ x, amplitude: amplitude * set * shoal * (1 - collapse * 0.86) * smooth((x - 66) / 66), breaking, collapse });
    }
    this.lip.fill(0);
    const tips: Point[] = [];
    for (let x = 0; x < 512; x++) {
      let elevation = 0, foamSource = 0;
      for (const w of waves) {
        const d = x - w.x;
        const front = 29 - w.breaking * 23, back = 46 + w.breaking * 6;
        const width = d < 0 ? front : back;
        elevation += w.amplitude * Math.exp(-(d * d) / (2 * width * width));
        elevation -= w.amplitude * 0.22 * Math.exp(-(((d + 56) / 31) ** 2)) * (1 - w.collapse);
        foamSource = Math.max(foamSource, w.collapse * Math.exp(-(((d + 10) / 32) ** 2)));
      }
      const fade = smooth((x - 83) / 70);
      const chop = settings.wind * 0.7 * Math.sin(x * 0.23 + t * 3.8) * fade;
      this.surface[x] = 165 - elevation * fade + chop;
      const wet = this.bed[x] > this.surface[x];
      this.wet[x] = wet ? 1 : this.wet[x] * Math.exp(-step * 0.04);
      const drift = Math.min(1, step * 26);
      this.nextFoam[x] = Math.max((this.foam[x] * (1-drift) + this.foam[Math.min(511,x+1)] * drift) * Math.exp(-step * 0.7), foamSource);
      if (wet && this.bed[x] - this.surface[x] < 3) this.nextFoam[x] = Math.max(this.nextFoam[x], 0.65);
    }
    [this.foam, this.nextFoam] = [this.nextFoam, this.foam];
    for (const w of waves) {
      if (w.breaking < 0.05 || w.collapse > 0.94 || w.amplitude < 5) continue;
      const crest = { x: w.x, y: 165 - w.amplitude };
      const reach = w.amplitude * 0.95 * w.breaking;
      const fall = w.amplitude * (0.08 + 0.98 * w.breaking ** 2);
      const outer: Point[] = [], inner: Point[] = [];
      const end = { x: crest.x - reach, y: crest.y + fall };
      // A finite-thickness water sheet curves over an air pocket (the barrel).
      for (let j = 0; j <= 40; j++) {
        const u = j / 40, v = 1-u;
        const x = v*v*v*crest.x + 3*v*v*u*(crest.x-reach*0.55) + 3*v*u*u*(crest.x-reach*1.2) + u*u*u*end.x;
        const y = v*v*v*crest.y + 3*v*v*u*(crest.y-w.amplitude*0.20) + 3*v*u*u*(crest.y-w.amplitude*0.16) + u*u*u*end.y;
        const thickness = (3 + w.amplitude * 0.14) * (1 - u * 0.65);
        outer.push({x,y}); inner.push({x:x+thickness*0.45,y:y+thickness});
      }
      this.sheet([...outer, ...inner.reverse()]); tips.push(end);
    }
    for (let y=0;y<256;y++) for(let x=0;x<512;x++) {
      const i=y*512+x,p=i*4,grain=this.grain[i],surface=this.surface[x],bed=this.bed[x];
      let r: number,g: number,b: number;
      if(y>=bed) {
        const below=y-bed,damp=this.wet[x]*Math.exp(-below/12),texture=(grain-.5)*12;
        r=222-damp*43+texture;g=202-damp*38+texture;b=157-damp*26+texture;
        const strata=Math.sin(below*.14+Math.sin(x/66)*.7)*2;r+=strata;g+=strata;b+=strata;
        if(below<1){r+=9;g+=9;b+=8;}if(grain>.996){r-=22;g-=21;b-=17;}
      } else if(y>=surface || this.lip[i]) {
        const below=Math.max(0,y-surface),deep=clamp(below/110);
        r=36-deep*22;g=156-deep*78;b=159-deep*58;
        const seabed=clamp(1-(bed-y)/18)*.22;r+=(172-r)*seabed;g+=(182-g)*seabed;b+=(143-b)*seabed;
        const light=Math.sin(x*.071+y*.053+t*.8)*Math.sin(x*.023-y*.07+t*.6)*6*Math.exp(-below/60);
        r+=light;g+=light;b+=light;
        const froth=this.foam[x]*Math.exp(-below/9)*(.55+grain*.45);
        r+=(234-r)*froth;g+=(247-g)*froth;b+=(229-b)*froth;
        if(this.lip[i]) {
          r=58+grain*9;g=181+grain*12;b=173+grain*9;
          if(y===0 || !this.lip[i-512]){r=200;g=234;b=218;}
        } else if(below<1.5){r+=48;g+=39;b+=25;}
      } else {
        const sky=clamp(y/170),dither=(grain-.5)*2;
        r=168+sky*57+dither;g=204+sky*29+dither;b=213+sky*12+dither;
      }
      data[p]=r;data[p+1]=g;data[p+2]=b;data[p+3]=255;
    }
    const dot=(px:number,py:number,size:number,alpha=1)=>{
      for(let dy=0;dy<size;dy++)for(let dx=0;dx<size;dx++){
        const x=Math.round(px)+dx,y=Math.round(py)+dy;
        if(x<0||x>=512||y<0||y>=256||y>=this.bed[x])continue;
        const p=(y*512+x)*4;data[p]+=(237-data[p])*alpha;data[p+1]+=(247-data[p+1])*alpha;data[p+2]+=(233-data[p+2])*alpha;
      }
    };
    this.emission+=step*55;const emit=Math.floor(this.emission);this.emission-=emit;
    for(const tip of tips)for(let n=0;n<emit&&this.spray.length<650;n++)this.spray.push({
      x:tip.x,y:tip.y,vx:-12-this.random()*(18+settings.wind*24),vy:-5-this.random()*15,life:.5+this.random()*.7,
    });
    for(const w of waves)if(w.collapse>.1&&w.collapse<.9)for(let n=0;n<emit;n++)this.spray.push({
      x:w.x-15+this.random()*20,y:165-w.amplitude,vx:-12-this.random()*36,vy:-15-this.random()*42*w.collapse,life:.5+this.random()*.5,
    });
    this.spray=this.spray.filter(p=>{
      p.x+=p.vx*step;p.vy+=95*step;p.y+=p.vy*step;p.life-=step;
      const x=Math.round(p.x);
      if(p.life<=0||x<0||x>=512||p.y>=this.bed[x])return false;
      if(p.y>this.surface[x]+3){this.foam[x]=Math.min(1,this.foam[x]+.3);return false;}
      dot(p.x,p.y,p.life>.7?2:1,Math.min(1,p.life*3));return true;
    });
    this.ctx.putImageData(this.image,0,0);
  }
}
