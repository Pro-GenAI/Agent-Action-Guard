import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

test('malformed request targets return 400 and do not terminate the HTTP process', () => {
	const source = `
import assert from 'node:assert/strict';
import { connect } from 'node:net';
import { createApiServer } from './dist/api-server.js';
let calls = 0;
const server = createApiServer({classifyActions: async (actions) => {calls++; return actions.map(() => ({label:null,confidence:.9}));}});
await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
try {
 const port=server.address().port;
 for (const target of ['http://[', 'http://%', 'http://[::1']) {
  const response=await new Promise((resolve,reject)=>{
   let text='';
   const client=connect(port,'127.0.0.1',()=>client.write('GET '+target+' HTTP/1.1\\r\\nHost: localhost\\r\\nConnection: close\\r\\n\\r\\n'));
   client.setTimeout(2000,()=>client.destroy(new Error('request timed out')));
   client.on('data',chunk=>text+=chunk);client.on('error',reject);client.on('end',()=>resolve(text));
  });
  assert.match(response,/HTTP\\/1.1 400/);
  assert.match(response,/Invalid request target/);
 }
 assert.equal(calls,0);
 assert.equal((await fetch('http://127.0.0.1:'+port+'/health')).status,200);
 const response=await fetch('http://127.0.0.1:'+port+'/v1/classify',{method:'POST',body:JSON.stringify({action:{}})});
 assert.equal(response.status,200);assert.equal(calls,1);
} finally {server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}
`;
	const result = spawnSync(
		process.execPath,
		['--input-type=module', '-e', source],
		{ encoding: 'utf8', timeout: 10000 },
	);
	assert.equal(result.error, undefined);
	assert.equal(result.status, 0, result.stderr);
});
