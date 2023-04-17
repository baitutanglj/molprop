# ###provide a test data to predict default model###
import json
with open('/mnt/home/linjie/projects/molprop/molprop/tests/example.json', 'rb') as f:
    mol_in = json.load(f)

from molprop.models.predict_model import predict
res_out = predict(mol_in=mol_in, model='AutoGluonModel', prop='LogF_Class', return_pred_prob=True)
print(res_out)
#{'name0': {'POS': 0.8706375360488892}, 'name1': 'NULL', 'name2': {'POS': 0.05391312763094902}, 'name3': {'POS': 0.7099388837814331}, 'name4': {'POS': 0.368745356798172}}

res_out = predict(mol_in=mol_in, model='AutoGluonModel', prop='LogF_Class', return_pred_prob=False)
print(res_out)
#{'name0': 'POS', 'name1': 'NULL', 'name2': 'NEG', 'name3': 'POS', 'name4': 'NEG'}

res_out = predict(mol_in=mol_in, model='AutoGluonModel', prop='HLM_Class', return_pred_prob=True)
print(res_out)
#{'name0': {'Medium': 0.5437608957290649, 'Low': 0.365691602230072, 'High': 0.09054741263389587}, 'name1': 'NULL', 'name2': {'Medium': 0.43076127767562866, 'Low': 0.43902307748794556, 'High': 0.13021565973758698}, 'name3': {'Medium': 0.16955414414405823, 'Low': 0.07128918915987015, 'High': 0.7591567039489746}, 'name4': {'Medium': 0.4264895021915436, 'Low': 0.4089065492153168, 'High': 0.16460394859313965}}

res_out = predict(mol_in=mol_in, model='AutoGluonModel', prop='HLM_Class', return_pred_prob=False)
print(res_out)
#{'name0': 'Medium', 'name1': 'NULL', 'name2': 'Low', 'name3': 'High', 'name4': 'Medium'}


res_out = predict(mol_in=mol_in, model='AutoGluonModel', prop='pLogS')
print(res_out)
#{'name0': 4.899094104766846, 'name1': 'NULL', 'name2': 4.900509834289551, 'name3': 3.422750473022461, 'name4': 7.986477851867676}





res_out = predict(mol_in=mol_in, model='CMPNN', prop='LogF_Class', return_pred_prob=True)
print(res_out)
#{'name0': {'POS': 0.6541534066200256}, 'name1': 'NULL', 'name2': {'POS': 0.34526661336421965}, 'name3': {'POS': 0.3460442513227463}, 'name4': {'POS': 0.6571748375892639}, 'name5': {'POS': 0.6420342326164246}}

res_out = predict(mol_in=mol_in, model='CMPNN', prop='LogF_Class', return_pred_prob=False)
print(res_out)
#{'name0': 'POS', 'name1': 'NULL', 'name2': 'NEG', 'name3': 'NEG', 'name4': 'POS', 'name5': 'POS'}

res_out = predict(mol_in=mol_in, model='CMPNN', prop='HLM_Class', return_pred_prob=True)
print(res_out)
#{'name0': {'Medium': 0.16925275177118237, 'Low': 0.7700589686632157, 'High': 0.06068828990322894}, 'name1': 'NULL', 'name2': {'Medium': 0.29053529910743237, 'Low': 0.7046988102607429, 'High': 0.0047658922597562745}, 'name3': {'Medium': 0.30346617255127056, 'Low': 0.6935961090028286, 'High': 0.0029377391861922847}, 'name4': {'Medium': 0.6134739906527102, 'Low': 0.07602386064900202, 'High': 0.3105021703485008}, 'name5': {'Medium': 0.5799998462200164, 'Low': 0.26866946974769235, 'High': 0.15133067372662481}}

res_out = predict(mol_in=mol_in, model='CMPNN', prop='HLM_Class', return_pred_prob=False)
print(res_out)
#{'name0': 'Low', 'name1': 'NULL', 'name2': 'Low', 'name3': 'Low', 'name4': 'Medium', 'name5': 'Medium'}

res_out = predict(mol_in=mol_in, model='CMPNN', prop='pLogS')
print(res_out)
#{'name0': 4.144638529891276, 'name1': 'NULL', 'name2': 3.5956422191007875, 'name3': 3.3886712104873817, 'name4': 4.003941251114772, 'name5': 4.559369284222119}






