// Run with: node --test tests/import-preparation-classes.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, writeFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { chromium } from 'playwright';

test('loaded preparation classes register in dataset order and mismatches remain seeds', async () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const harness = path.resolve(root, `.preparation-${process.pid}.local`);
  assert.equal(path.dirname(harness), root);
  await mkdir(harness);
  await writeFile(path.join(harness, 'index.html'), '<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness, 'main.tsx'), `import React from 'react';
    import {createRoot} from 'react-dom/client';
    import {Track2ImportCheckpointPanel} from '../src/components/Track2ImportCheckpointPanel';
    import '../src/index.css';
    createRoot(document.getElementById('root')!).render(<Track2ImportCheckpointPanel/>);`);
  let server, browser;
  try {
    server = await createServer({ root, server: { host: '127.0.0.1', port: 0 } });
    await server.listen();
    browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE });
    const page = await browser.newPage();
    const classes = { 2: 'person', 0: 'helmet', 1: 'gloves' };
    let preparation = { name: 'helmet_test_DP', config: { dataset_assemble: { label_classes: classes } } };
    let savedMode = false;
    let fileClasses = { 0: 'person', 1: 'car' };
    const saves = [], registrations = [], detectors = [];
    await page.route('**/track2/**', async route => {
      const req = route.request(), url = new URL(req.url()), body = req.postDataJSON();
      let response = {};
      if (url.pathname === '/track2/pull-control') response = { config: {
        training_data_preparation: preparation,
        card_state: { 'import a pretrained coco or an already-trained model': true },
        import_checkpoint: { architecture: 'yolo26s', read_key: 'test.pt', use_file_classes: !savedMode, use_preparation_classes: savedMode },
      } };
      else if (url.pathname.endsWith('/config')) { saves.push(body); response = { config: body }; }
      else if (url.pathname.endsWith('/architectures')) response = { architectures: ['yolo26s'] };
      else if (url.pathname.endsWith('/library')) response = { weights: [] };
      else if (url.pathname.endsWith('/verify')) response = { key: 'test.pt', classes: fileClasses, detected_architecture: 'yolo26s', size_bytes: 1024 };
      else if (url.pathname.endsWith('/detectors')) {
        if (req.method() === 'POST') { detectors.push(body); response = { slug: body.slug }; }
        else response = { detectors: [] };
      } else if (url.pathname.endsWith('/register')) {
        registrations.push(body);
        response = { model_package_arn: 'test-arn', detector_slug: body.detector_slug, architecture: 'yolo26s' };
      } else if (url.pathname.endsWith('/metrics')) response = { rows: [] };
      await route.fulfill({ json: response });
    });
    const url = `http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`;
    const option = () => page.getByRole('checkbox', { name: /Use classes from the loaded Data Preparation file/ });
    await page.goto(url);
    const savedChoice = page.waitForRequest(req => req.url().endsWith('/config')
      && req.postDataJSON()?.import_checkpoint?.use_preparation_classes === true);
    await option().check();
    assert.equal((await savedChoice).postDataJSON().import_checkpoint.use_file_classes, false);
    assert.equal(await page.getByRole('checkbox', { name: /^Use the classes in this file/ }).isChecked(), false);
    await page.getByText('0: helmet', { exact: true }).waitFor();
    await page.getByText('2: person', { exact: true }).waitFor();
    await page.getByText(/It will register as a training seed/).waitFor();
    if (process.env.PREPARATION_SCREENSHOT) {
      await option().scrollIntoViewIfNeeded();
      await page.screenshot({ path: process.env.PREPARATION_SCREENSHOT });
    }
    await page.getByPlaceholder('Name, e.g. Forklift').fill('Prepared Model');
    const firstRegistration = page.waitForResponse('**/base-models/register');
    await page.getByRole('button', { name: 'Register New Model Name', exact: true }).click();
    await firstRegistration;
    assert.deepEqual(detectors.at(-1).classes, { 0: 'helmet', 1: 'gloves', 2: 'person' });
    assert.equal(registrations.at(-1).allow_class_mismatch, true);
    assert.deepEqual(registrations.at(-1).keep_classes, []);

    // Restore the saved option; matching checkpoint classes need no mismatch exception.
    savedMode = true;
    fileClasses = { 0: 'helmet', 1: 'gloves', 2: 'person' };
    await page.reload();
    await page.getByText('0: helmet', { exact: true }).waitFor();
    assert.equal(await option().isChecked(), true);
    await page.getByPlaceholder('Name, e.g. Forklift').fill('Matching Model');
    const registered = page.waitForResponse('**/base-models/register');
    await page.getByRole('button', { name: 'Register New Model Name', exact: true }).click();
    await registered;
    assert.equal(registrations.at(-1).allow_class_mismatch, false);

    // Missing or invalid preparation metadata must never register an empty model.
    for (const invalid of [undefined, { name: 'empty_DP', config: {} },
      { name: 'duplicate_DP', config: { dataset_assemble: { label_classes: { 0: 'person', 1: 'person' } } } }]) {
      preparation = invalid;
      await page.reload();
      await option().waitFor();
      assert.equal(await option().isDisabled(), true);
      await page.getByPlaceholder('Name, e.g. Forklift').fill('Invalid Model');
      assert.equal(await page.getByRole('button', { name: 'Register New Model Name', exact: true }).isDisabled(), true);
    }
  } finally {
    await browser?.close();
    await server?.close();
    await rm(harness, { recursive: true, force: true });
  }
});
