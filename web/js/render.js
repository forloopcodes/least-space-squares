// Renderer for packings: WebGL2 instanced quads (one draw call for any n) with a Canvas2D fallback.
const VS = `#version 300 es
in vec2 aQuad; in vec3 aInst;
uniform vec2 uScale; uniform vec2 uOffset;
out vec2 vLocal; flat out float vAngle;
void main() {
  float c = cos(aInst.z), s = sin(aInst.z);
  vec2 p = aInst.xy + vec2(c * aQuad.x - s * aQuad.y, s * aQuad.x + c * aQuad.y);
  gl_Position = vec4(p * uScale + uOffset, 0.0, 1.0);
  vLocal = aQuad; vAngle = aInst.z;
}`;
const FS = `#version 300 es
precision highp float;
in vec2 vLocal; flat in float vAngle;
uniform float uBorder;
out vec4 o;
void main() {
  float a = abs(sin(2.0 * vAngle));
  vec3 fill = a < 1e-6 ? vec3(0.70, 0.80, 0.89) : (abs(a - 1.0) < 1e-6 ? vec3(0.98, 0.71, 0.68) : vec3(0.80, 0.92, 0.77));
  float e = 0.5 - max(abs(vLocal.x), abs(vLocal.y));
  o = vec4(e < uBorder ? vec3(0.1) : fill, 1.0);
}`;
const OUT_VS = `#version 300 es
in vec2 aPos; uniform vec2 uScale; uniform vec2 uOffset;
void main() { gl_Position = vec4(aPos * uScale + uOffset, 0.0, 1.0); }`;
const OUT_FS = `#version 300 es
precision highp float; out vec4 o; void main() { o = vec4(0.0, 0.0, 0.0, 1.0); }`;

function compile(gl, vs, fs) {
  const prog = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    gl.attachShader(prog, sh);
  }
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  return prog;
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.s = 1; this.n = 0; this.zoom = 1; this.cx = 0.5; this.cy = 0.5;
    this.gl = canvas.getContext('webgl2', { antialias: true, alpha: false, preserveDrawingBuffer: true });
    this.mode = this.gl ? 'webgl2' : 'canvas2d';
    if (this.gl) this._initGL(); else this.ctx = canvas.getContext('2d');
    this._bindInteraction();
  }
  _initGL() {
    const gl = this.gl;
    this.prog = compile(gl, VS, FS);
    this.outProg = compile(gl, OUT_VS, OUT_FS);
    this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
    const quad = new Float32Array([-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5]);
    this.quadBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf); gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    const aQuad = gl.getAttribLocation(this.prog, 'aQuad'); gl.enableVertexAttribArray(aQuad); gl.vertexAttribPointer(aQuad, 2, gl.FLOAT, false, 0, 0);
    this.instBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    const aInst = gl.getAttribLocation(this.prog, 'aInst'); gl.enableVertexAttribArray(aInst); gl.vertexAttribPointer(aInst, 3, gl.FLOAT, false, 0, 0); gl.vertexAttribDivisor(aInst, 1);
    gl.bindVertexArray(null);
    this.outVao = gl.createVertexArray(); gl.bindVertexArray(this.outVao);
    this.outBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.outBuf);
    const aPos = gl.getAttribLocation(this.outProg, 'aPos'); gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.uScale = gl.getUniformLocation(this.prog, 'uScale'); this.uOffset = gl.getUniformLocation(this.prog, 'uOffset'); this.uBorder = gl.getUniformLocation(this.prog, 'uBorder');
    this.oScale = gl.getUniformLocation(this.outProg, 'uScale'); this.oOffset = gl.getUniformLocation(this.outProg, 'uOffset');
  }
  setPacking(s, squares) {
    this.s = s; this.n = squares.length / 3; this.squares = squares;
    if (this.gl) {
      const gl = this.gl;
      this.inst = squares instanceof Float32Array ? squares : Float32Array.from(squares);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf); gl.bufferData(gl.ARRAY_BUFFER, this.inst, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.outBuf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, s, 0, s, s, 0, s]), gl.DYNAMIC_DRAW);
    }
    this.fit();
  }
  fit() { this.zoom = 1; this.cx = this.s / 2; this.cy = this.s / 2; this.draw(); }
  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr)), h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    return [w, h];
  }
  /** pixels per world unit */
  ppu(w, h) { return Math.min(w, h) * 0.94 * this.zoom / this.s; }
  draw() {
    const [w, h] = this._resize();
    if (this.n === 0 && !this.squares) return;
    const ppu = this.ppu(w, h);
    if (this.gl) {
      const gl = this.gl;
      gl.viewport(0, 0, w, h); gl.clearColor(1, 1, 1, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      const sx = 2 * ppu / w, sy = 2 * ppu / h;
      gl.useProgram(this.prog); gl.bindVertexArray(this.vao);
      gl.uniform2f(this.uScale, sx, sy); gl.uniform2f(this.uOffset, -this.cx * sx, -this.cy * sy);
      gl.uniform1f(this.uBorder, Math.min(0.08, 1.2 / ppu));
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.n);
      gl.useProgram(this.outProg); gl.bindVertexArray(this.outVao);
      gl.uniform2f(this.oScale, sx, sy); gl.uniform2f(this.oOffset, -this.cx * sx, -this.cy * sy);
      gl.drawArrays(gl.LINE_LOOP, 0, 4);
      gl.bindVertexArray(null);
    } else {
      const ctx = this.ctx; ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, h);
      ctx.setTransform(ppu, 0, 0, -ppu, w / 2 - this.cx * ppu, h / 2 + this.cy * ppu);
      ctx.lineWidth = 1 / ppu; ctx.strokeStyle = '#000'; ctx.strokeRect(0, 0, this.s, this.s);
      const sq = this.squares, lw = Math.min(0.08, 1.2 / ppu);
      ctx.lineWidth = lw;
      for (let i = 0; i < this.n; i++) {
        const x = sq[3 * i], y = sq[3 * i + 1], t = sq[3 * i + 2], a = Math.abs(Math.sin(2 * t));
        ctx.fillStyle = a < 1e-6 ? '#b3cde3' : (Math.abs(a - 1) < 1e-6 ? '#fbb4ae' : '#ccebc5');
        ctx.save(); ctx.translate(x, y); ctx.rotate(t); ctx.fillRect(-0.5, -0.5, 1, 1); ctx.strokeStyle = '#1a1a1a'; ctx.strokeRect(-0.5, -0.5, 1, 1); ctx.restore();
      }
    }
  }
  _bindInteraction() {
    const c = this.canvas; let drag = null;
    c.addEventListener('wheel', e => {
      e.preventDefault();
      const f = Math.exp(-e.deltaY * 0.0015);
      const [w, h] = [c.width, c.height], ppu = this.ppu(w, h), dpr = window.devicePixelRatio || 1;
      const mx = (e.offsetX * dpr - w / 2) / ppu + this.cx, my = -(e.offsetY * dpr - h / 2) / ppu + this.cy;
      this.zoom = Math.min(1e4, Math.max(0.2, this.zoom * f));
      const ppu2 = this.ppu(w, h);
      this.cx = mx - (e.offsetX * dpr - w / 2) / ppu2; this.cy = my + (e.offsetY * dpr - h / 2) / ppu2;
      this.draw();
    }, { passive: false });
    c.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, cx: this.cx, cy: this.cy }; c.setPointerCapture(e.pointerId); });
    c.addEventListener('pointermove', e => {
      if (!drag) return;
      const dpr = window.devicePixelRatio || 1, ppu = this.ppu(c.width, c.height);
      this.cx = drag.cx - (e.clientX - drag.x) * dpr / ppu; this.cy = drag.cy + (e.clientY - drag.y) * dpr / ppu; this.draw();
    });
    c.addEventListener('pointerup', () => { drag = null; });
    c.addEventListener('dblclick', () => this.fit());
    new ResizeObserver(() => this.draw()).observe(c);
  }
}
