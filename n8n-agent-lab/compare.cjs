const assert = require('node:assert/strict');
function call(file,name,input) {
  const node=require(file).nodes.find(n=>n.name===name);
  return JSON.parse(Function('query',node.parameters.jsCode)(input));
}
const original='./returns-desk.json', byId='./returns-desk-by-id.json';
const order=call(original,'lookup_order','DEMO-1044');
const altered=call(original,'return_policy',JSON.stringify({...order,sealed:true}));
const bound=call(byId,'check_return',order.id);
assert.equal(altered.eligible,true);
assert.deepEqual(bound,{id:'DEMO-1044',version:'RET-3',status:'evaluated',eligible:false,reason:'seal_open'});
assert.equal(call(byId,'check_return',JSON.stringify({...order,sealed:true})).status,'invalid_id');
console.log(JSON.stringify({record:order,alteredInputResult:altered,checkedById:bound},null,2));
