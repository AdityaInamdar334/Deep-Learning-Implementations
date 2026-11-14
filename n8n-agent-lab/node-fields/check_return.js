const id = String(query).trim().toUpperCase();
if (id === 'SUMMARY') return JSON.stringify({version:'RET-3',rule:'Standard items: within 30 days inclusive. Hygiene items also require an intact seal.'});
const lookup = (query) => {const id = String(query).trim().toUpperCase();
const orders = {
  'DEMO-1042': {itemClass:'standard', daysSinceDelivery:30, sealed:false},
  'DEMO-1043': {itemClass:'standard', daysSinceDelivery:31, sealed:true},
  'DEMO-1044': {itemClass:'hygiene', daysSinceDelivery:4, sealed:false},
  'DEMO-1045': {itemClass:'hygiene', daysSinceDelivery:4, sealed:true}
};
if (!/^DEMO-[0-9]{4}$/.test(id)) return JSON.stringify({status:'invalid_id'});
if (!Object.hasOwn(orders,id)) return JSON.stringify({status:'not_found',id});
return JSON.stringify({status:'found',id,...orders[id]});};
const policy = (query) => {const version = 'RET-3';
const raw = String(query).trim();
if (raw.toLowerCase() === 'summary') return JSON.stringify({version, rule:'Standard items: within 30 days inclusive. Hygiene items: same window and seal intact.'});
let f;
try { f = JSON.parse(raw); } catch { return JSON.stringify({version,status:'invalid_facts'}); }
if (!f || !['standard','hygiene'].includes(f.itemClass) ||
    !Number.isInteger(f.daysSinceDelivery) || f.daysSinceDelivery < 0 ||
    typeof f.sealed !== 'boolean') return JSON.stringify({version,status:'invalid_facts'});
const reason = f.daysSinceDelivery > 30 ? 'outside_window' :
    f.itemClass === 'hygiene' && !f.sealed ? 'seal_open' : 'within_policy';
return JSON.stringify({version,status:'evaluated',eligible:reason==='within_policy',reason});};
const record = JSON.parse(lookup(id));
if (record.status !== 'found') return JSON.stringify(record);
const result = JSON.parse(policy(JSON.stringify(record)));
return JSON.stringify({id:record.id,...result});
