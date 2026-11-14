// Runs only this lab's trusted, hand-authored tool code.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const wf=require('./returns-desk.json');
function run(name,input){
  const node=wf.nodes.find(n=>n.name===name);
  return JSON.parse(Function('query',node.parameters.jsCode)(input));
}
test('the two actual tool nodes connect to the agent',()=>{
  for(const name of ['lookup_order','return_policy'])
    assert.equal(wf.connections[name].ai_tool[0][0].node,'Returns analyst');
});
for(const [id,eligible,reason] of [
  ['DEMO-1042',true,'within_policy'],['DEMO-1043',false,'outside_window'],
  ['DEMO-1044',false,'seal_open'],['DEMO-1045',true,'within_policy']]){
  test(id,()=>{
    const order=run('lookup_order',id); assert.equal(order.status,'found');
    const result=run('return_policy',JSON.stringify(order));
    assert.deepEqual(result,{version:'RET-3',status:'evaluated',eligible,reason});
  });
}
test('unknown, malformed and normalized IDs',()=>{
 assert.equal(run('lookup_order','DEMO-9999').status,'not_found');
 assert.equal(run('lookup_order','all').status,'invalid_id');
 assert.equal(run('lookup_order',' demo-1042 ').id,'DEMO-1042');
});
test('summary is available without an order',()=>assert.match(run('return_policy','summary').rule,/inclusive/));
test('bad and subtly mistyped input fails closed',()=>{
 for (const input of ['null','bad','{}','{"itemClass":"standard","daysSinceDelivery":-1,"sealed":true}','{"itemClass":"hygiene","daysSinceDelivery":1,"sealed":"false"}'])
   assert.equal(run('return_policy',input).status,'invalid_facts');
});
