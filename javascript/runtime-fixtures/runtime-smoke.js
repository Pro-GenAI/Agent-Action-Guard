import assert from 'node:assert/strict';

import {
	ActionGuardDecision,
	actionGuarded,
	ensureActionSafety,
	flattenActionToText,
	isActionHarmful,
} from '../src/index.js';

assert.equal(ActionGuardDecision.ALLOW, 'ALLOW');

assert.equal(
	flattenActionToText({
		type: 'function',
		function: {
			name: 'send_email',
			arguments: { to: 'user@example.com', subject: 'Hello' },
		},
	}),
	'Call function send_email: to=user@example.com, subject=Hello',
);

const harmlessClassifier = {
	predict: async () => ({ label: 'safe', confidence: 0.99 }),
};

const harmfulClassifier = {
	predict: async () => ({ label: 'harmful', confidence: 0.91 }),
};

assert.deepEqual(
	await isActionHarmful(
		{ type: 'function', function: { name: 'ping', arguments: {} } },
		harmlessClassifier,
	),
	{ label: null, confidence: 0.99 },
);

assert.equal(
	await ensureActionSafety(
		{ type: 'function', function: { name: 'ping', arguments: {} } },
		{ actionClassifier: harmlessClassifier },
	),
	true,
);

const guarded = actionGuarded(
	async function sendEmail(params) {
		return params.to;
	},
	{
		confThreshold: 0.5,
		actionClassifier: harmfulClassifier,
	},
);

await assert.rejects(guarded({ to: 'user@example.com' }), /harmful/);
