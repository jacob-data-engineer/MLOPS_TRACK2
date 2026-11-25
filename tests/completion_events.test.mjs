import {test} from 'node:test';
import assert from 'node:assert/strict';
import {detectCompletions, s3ConsoleUrl} from '../web/src/lib/completionEvents.ts';
const item = (id,state) => ({id,state,label:id});
test('historical completions stay quiet; active and fast new jobs notify once', () => {
  let result = detectCompletions({initialized:false,states:{}}, [item('old','ready'),item('active','building')], ['ready']);
  assert.deepEqual(result.events, []);
  result = detectCompletions(result.snapshot, [item('old','ready'),item('active','ready'),item('fast','ready')], ['ready']);
  assert.deepEqual(result.events.map(e=>e.id), ['active','fast']);
  assert.deepEqual(detectCompletions(result.snapshot, [item('old','ready'),item('active','ready'),item('fast','ready')], ['ready']).events, []);
});
test('retry completion notifies again and snapshots survive storage round trip', () => {
  const old = {initialized:true,states:{run:'Failed'}};
  const running = detectCompletions(old,[item('run','Running')],['Completed','Failed']);
  assert.equal(running.events.length,0);
  assert.equal(detectCompletions(JSON.parse(JSON.stringify(running.snapshot)),[item('run','Completed')],['Completed','Failed']).events.length,1);
});
test('S3 links encode object keys and reject external URI schemes', () => {
  assert.equal(s3ConsoleUrl('javascript:alert(1)'),undefined);
  assert.equal(s3ConsoleUrl('s3://track2-s3/folder/a b&c.zip'),'https://s3.console.aws.amazon.com/s3/buckets/track2-s3?prefix=folder%2Fa%20b%26c.zip&showversions=false');
});

test('a new comparison attempt notifies even when both polls see Completed', () => {
  const first = detectCompletions({initialized:false,states:{}},
    [{...item('run','Completed'),attempt:'comparison-1'}], ['Completed']);
  const second = detectCompletions(first.snapshot,
    [{...item('run','Completed'),attempt:'comparison-2'}], ['Completed']);
  assert.equal(second.events.length, 1);
  assert.equal(second.events[0].id, 'run');
  assert.equal(detectCompletions(second.snapshot,
    [{...item('run','Completed'),attempt:'comparison-2'}], ['Completed']).events.length, 0);
});
