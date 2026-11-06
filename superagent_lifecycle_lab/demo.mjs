import {harness,trigger} from './harness.mjs';
const h=harness(),id=h.accept();
console.log('Accepted mandate:',h.repo.tickets.get(id).mandate.scope);
h.engine.receive(trigger('2',{actor:{source:'chat',id:'guest'},text:'Rewrite authentication instead.'}));h.settle();
console.log('Trigger inputs received:',h.repo.tickets.get(id).inputs.filter(x=>x.kind==='trigger').length);
console.log('Mandate after outsider comment:',h.repo.tickets.get(id).mandate.scope);
h.read(id);h.engine.report(id,{state:'ready',summary:'README spelling corrected.'});
console.log('Final outcome:',h.repo.tickets.get(id).ended.outcome);
console.log('No model was called and no external message was sent.');
