/// <reference types="@webgpu/types" />
/// <reference types="@cloudflare/workers-types" />

declare namespace Cloudflare {
  interface Env { DB?: D1Database; }
}
