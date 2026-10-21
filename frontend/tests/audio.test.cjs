const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('typescript');
const source = fs.readFileSync(require('node:path').join(__dirname, '../src/lib/audio.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
let closed = 0;
const context = {
  exports: {}, Blob, DOMException,
  AudioContext: class {
    async decodeAudioData() {
      return { length: 3, sampleRate: 22050, numberOfChannels: 2,
        getChannelData: () => new Float32Array([-1, 0, 1]) };
    }
    async close() { closed++; }
  },
};
vm.runInNewContext(compiled, context);
test('compressed input becomes PCM WAV with correct header and samples', async () => {
  const wav = await context.exports.toWav(new Blob(['compressed'], { type: 'audio/webm' }));
  const bytes = Buffer.from(await wav.arrayBuffer());
  assert.equal(wav.type, 'audio/wav');
  assert.equal(bytes.toString('ascii', 0, 4), 'RIFF');
  assert.equal(bytes.toString('ascii', 8, 12), 'WAVE');
  assert.equal(bytes.readUInt32LE(4), bytes.length - 8);
  assert.equal(bytes.readUInt16LE(20), 1);
  assert.equal(bytes.readUInt16LE(22), 1);
  assert.equal(bytes.readUInt32LE(24), 22050);
  assert.equal(bytes.readInt16LE(44), -32768);
  assert.equal(bytes.readInt16LE(46), 0);
  assert.equal(bytes.readInt16LE(48), 32767);
  assert.equal(closed, 1);
});
test('empty and oversized input are rejected', async () => {
  await assert.rejects(context.exports.toWav(new Blob([])), /empty/);
  await assert.rejects(context.exports.toWav(new Blob([new Uint8Array(25 * 1024 * 1024 + 1)])), /25 MB/);
});
