"""Make 36 synthetic mixed-format docs and 36 relevance-labeled queries."""
from pathlib import Path
import json
from textwrap import wrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
ROOT=Path(__file__).parent; DOC=ROOT/'data/demo';DOC.mkdir(parents=True,exist_ok=True)
records=[
('cedar_release','Cedar release checklist','The production deployment window begins on Thursday at 21:30 UTC. The rollback owner is Maya Chen. Cedar deployments require two approvals.','When does the Cedar production deployment window begin?','exact'),
('maple_backup','Maple backup runbook','The encrypted archive is retained for 45 days. Restores require a signed ticket. Nightly backups finish at 02:00 UTC.','How long does the Maple archive remain available?','paraphrase'),
('birch_api','Birch API documentation','The request limit is 240 calls per minute. Authorization uses a bearer token. Exhausted clients receive HTTP 429.','What is Birch API request rate limit?','exact'),
('oak_roadmap','Oak roadmap','The first accessibility audit is scheduled for March 18. The roadmap includes keyboard navigation and contrast remediation.','When is the Oak accessibility audit scheduled?','exact'),
('pine_oncall','Pine on-call handbook','Page the reliability engineer after three failed health probes. Escalation goes to the amber duty group.','Who is alerted after three failed Pine health checks?','paraphrase'),
('ash_legal','Ash legal review','Contracts must be retained for seven years after expiration. Legal questions go to the central counsel queue.','What is the Ash contract retention duration?','paraphrase'),
('quartz_invoice','Quartz invoice guide','Invoices under 500 dollars can be approved by the team lead. Larger invoices require finance signoff.','Who can sign off on small Quartz invoices?','paraphrase'),
('nebula_research','Nebula research notes','The pilot study enrolled 48 participants. The main outcome is mean response latency.','How many people participated in the Nebula pilot?','paraphrase'),
('delta_training','Delta training plan','The onboarding workshop is held every second Tuesday. New hires receive laptops on day one.','What day is the Delta new-hire workshop held?','exact'),
('atlas_database','Atlas database operations','The warehouse stores snapshots in the north-region bucket. Schema migrations run after 18:00 UTC.','Where are Atlas warehouse snapshots stored?','paraphrase'),
('ember_incident','Ember incident report','The root cause was an expired TLS certificate on the internal gateway. Recovery took 37 minutes.','What caused the Ember internal service outage?','paraphrase'),
('juniper_security','Juniper security policy','Multi-factor authentication is required for administrator accounts. Hardware keys are the preferred second factor.','Does Juniper require another verification factor for admins?','paraphrase'),
('saffron_budget','Saffron budget','The quarterly cloud budget is 18,500 dollars. The expense owner is the infrastructure director.','What is the quarterly Saffron cloud spending limit?','paraphrase'),
('cobalt_design','Cobalt design notes','The header uses a 12-column grid. The primary interface font is Inter.','What font is specified for the Cobalt interface?','exact'),
('violet_tests','Violet testing strategy','Smoke tests run before every production push. Nightly integration tests begin at 01:00 UTC.','When do Violet integration tests execute?','paraphrase'),
('coral_recruiting','Coral hiring notes','The interview loop has four stages. The take-home exercise has been removed.','How many phases are there in Coral interviews?','paraphrase'),
('lunar_storage','Lunar storage runbook','Cold storage data is replicated in three regions. Restore requests must use the operations portal.','How many locations hold Lunar cold backups?','paraphrase'),
('nova_analytics','Nova analytics guide','Session events are kept for 90 days. Dashboard refreshes occur every fifteen minutes.','How often does the Nova dashboard update?','paraphrase'),
('orbit_mobile','Orbit mobile release','Minimum supported Android version is 12. The next beta build is distributed through internal testing.','What is the oldest Android version supported by Orbit?','paraphrase'),
('fable_review','Fable code review','Two engineers must review production database changes. A single reviewer is allowed for documentation-only updates.','How many reviewers approve Fable database modifications?','paraphrase'),
('raven_alert','Raven alerting guide','A high-severity alert pages the primary responder immediately. Low-severity notifications are collected into a daily digest.','What happens to low-priority Raven alerts?','paraphrase'),
('mint_support','Mint support handbook','First-response target for standard cases is four business hours. Urgent incidents require a reply within thirty minutes.','What is the first reply target for normal Mint support tickets?','paraphrase'),
('opal_search','Opal search experiment','The lexical baseline uses BM25. The hybrid ranking uses reciprocal rank fusion with rank constant 60.','Which method combines Opal search rankings?','paraphrase'),
('topaz_pipeline','Topaz pipeline specification','The training dataset is versioned with DVC. Metadata is tracked in a SQLite manifest.','Which tool versions the Topaz training data?','exact'),
('aurora_meetings','Aurora meeting notes','The weekly synchronization starts at 10:15 on Mondays. Action items are stored in a shared Markdown document.','When does the Aurora team hold its weekly sync?','paraphrase'),
('silver_compliance','Silver compliance register','Audit evidence must be exported as PDF/A. The reviewer deadline is the ninth business day.','Which document format is required for Silver audit exports?','paraphrase'),
('helios_gpu','Helios GPU benchmark','The evaluation uses batch size 1. Decode throughput is reported in output tokens per second.','What batch size does the Helios benchmark use?','exact'),
('tundra_data','Tundra data dictionary','All timestamps are stored in UTC. Null identifiers are rejected during ingestion.','Which timezone is used for Tundra timestamps?','exact'),
('pebble_release','Pebble release notes','The library added pagination support in version 2.4. Deprecated flags are removed in version 3.0.','Which Pebble version introduced pagination?','paraphrase'),
('spruce_incident','Spruce outage postmortem','The incident lasted 22 minutes. An overloaded connection pool caused request failures.','What component triggered the Spruce request errors?','paraphrase'),
('harbor_guide','Harbor deployment guide','Blue-green deployments switch traffic after smoke tests pass. The old environment remains online for 20 minutes.','How long is the old Harbor environment retained after a swap?','paraphrase'),
('glacier_notes','Glacier research diary','The team used Spearman correlation because the data was non-normal. Analysis scripts are stored in the experiments directory.','Why did Glacier researchers choose Spearman correlation?','paraphrase'),
('meadow_policy','Meadow work policy','Travel expense claims must be submitted within 14 calendar days. Approval belongs to the cost-center owner.','How soon must Meadow employees submit travel expenses?','paraphrase'),
('cinder_api','Cinder API changes','The retry-after header contains seconds. Requests without an idempotency key may be duplicated after retries.','What units does the Cinder retry-after header use?','paraphrase'),
('willow_ml','Willow ML notebook','The baseline classifier is logistic regression. Training uses stratified five-fold validation.','How is the Willow classifier evaluated across folds?','paraphrase'),
('onyx_notes','Onyx meeting summary','The design freeze is Friday at noon. The escalation contact is the platform team.','When is the Onyx design freeze?','exact')]
queries=[]
for i,(stem,title,body,query,category) in enumerate(records):
    ext=['.md','.txt','.pdf'][i%3]
    path=DOC/(stem+ext)
    text=f'{title}\n\n{body}\n\nThis note is part of the fictional internal knowledge-base demonstration. It is not a real company policy.'
    if ext=='.pdf':
        c=canvas.Canvas(str(path),pagesize=letter); y=745
        for line in [title,'']+wrap(body,92)+['','Fictional benchmark document.']:
            c.drawString(48,y,line);y-=20
        c.showPage();c.save()
    else:path.write_text(('# '+title+'\n\n' if ext=='.md' else title+'\n\n')+body+'\n',encoding='utf-8')
    queries.append(dict(id=f'Q{i+1:02}',query=query,relevant_paths=[path.name],category=category))
(ROOT/'data/queries.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in queries))
print('Created',len(records),'documents and',len(queries),'labeled queries')
