import { copyFile, mkdir } from 'node:fs/promises';
import { URL } from 'node:url';

await mkdir(new URL('../dist/', import.meta.url), { recursive: true });
await copyFile(
  new URL('../src/action_classifier_model.onnx', import.meta.url),
  new URL('../dist/action_classifier_model.onnx', import.meta.url),
);
