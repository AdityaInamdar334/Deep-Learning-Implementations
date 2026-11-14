const id = String(query).trim().toUpperCase();
const orders = {
  'DEMO-1042': {itemClass:'standard', daysSinceDelivery:30, sealed:false},
  'DEMO-1043': {itemClass:'standard', daysSinceDelivery:31, sealed:true},
  'DEMO-1044': {itemClass:'hygiene', daysSinceDelivery:4, sealed:false},
  'DEMO-1045': {itemClass:'hygiene', daysSinceDelivery:4, sealed:true}
};
if (!/^DEMO-[0-9]{4}$/.test(id)) return JSON.stringify({status:'invalid_id'});
if (!Object.hasOwn(orders,id)) return JSON.stringify({status:'not_found',id});
return JSON.stringify({status:'found',id,...orders[id]});
