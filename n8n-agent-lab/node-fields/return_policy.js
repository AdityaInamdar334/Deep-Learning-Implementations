const version = 'RET-3';
const raw = String(query).trim();
if (raw.toLowerCase() === 'summary') return JSON.stringify({version, rule:'Standard items: within 30 days inclusive. Hygiene items: same window and seal intact.'});
let f;
try { f = JSON.parse(raw); } catch { return JSON.stringify({version,status:'invalid_facts'}); }
if (!f || !['standard','hygiene'].includes(f.itemClass) ||
    !Number.isInteger(f.daysSinceDelivery) || f.daysSinceDelivery < 0 ||
    typeof f.sealed !== 'boolean') return JSON.stringify({version,status:'invalid_facts'});
const reason = f.daysSinceDelivery > 30 ? 'outside_window' :
    f.itemClass === 'hygiene' && !f.sealed ? 'seal_open' : 'within_policy';
return JSON.stringify({version,status:'evaluated',eligible:reason==='within_policy',reason});
