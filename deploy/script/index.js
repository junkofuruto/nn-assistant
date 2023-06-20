import init from './assistant.js';

async function run() {
    const module = await init();
    module.main();
}

run();