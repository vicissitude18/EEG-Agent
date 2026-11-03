"""JSON schemas must have independent property nodes before adding constraints."""
import json
def isolated_schema(schema):return json.loads(json.dumps(schema))
def check_schema_isolation():
    shared={'type':'string'};number={'type':'number'}
    source={'candidate_id':shared,'reason':shared,'evidence_id':shared,'onset':number,'offset':number}
    result=isolated_schema(source);result['candidate_id']['enum']=['C1'];result['onset']['maximum']=10
    assert 'enum' not in result['reason'] and 'enum' not in result['evidence_id']
    assert 'maximum' not in result['offset'] and 'enum' not in source['candidate_id']
    print('SCHEMA_ISOLATION_PASS: candidate IDs, reasons, evidence IDs and numeric bounds are independent')
if __name__=='__main__':check_schema_isolation()
