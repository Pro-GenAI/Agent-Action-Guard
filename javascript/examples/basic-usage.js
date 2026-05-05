// import { isActionHarmful } from '../src/index.js';
import { isActionHarmful } from 'agent-action-guard';

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

	const { label, confidence } = await isActionHarmful(action);
	console.log('Decision:', label);
	console.log('Confidence:', confidence);
}

main().catch((error) => {
	console.error(error);
	process.exitCode = 1;
});
