export default [
	{
		files: ['**/*.{js,ts}'],
		ignores: ['node_modules/**', 'build/**', 'dist/**', 'coverage/**'],
		languageOptions: {
			ecmaVersion: 'latest',
			sourceType: 'module',
			globals: {
				console: 'readonly',
				process: 'readonly',
			},
		},
		rules: {
			'no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
			'no-undef': 'error',
		},
	},
];
