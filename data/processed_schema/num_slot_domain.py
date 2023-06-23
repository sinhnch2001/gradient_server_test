import json
schema = json.load(open("./schema_final.json"))
num_slot_domain = {}
for domain,slot in schema.items():
    num_slot_domain.setdefault(domain,len(slot))
with open("./num_slot_domain.json", 'w') as f:
    json.dump(num_slot_domain, f, indent=4)
