import { ensureActionSafety, isActionHarmful } from '../src/index.js';

async function main() {
	const action = {
		type: 'function',
		function: {
			name: 'send_email',
			arguments: {
				to: 'user@example.com',
				subject: 'Status update',
				body: 'Hello from Action Guard',
			},
		},
	};

	const decision = await isActionHarmful(action);
	console.log('Decision:', decision);

	const isSafe = await ensureActionSafety(action);
	console.log('Safe to execute:', isSafe);
}

main().catch((error) => {
	console.error(error);
	process.exitCode = 1;
});
