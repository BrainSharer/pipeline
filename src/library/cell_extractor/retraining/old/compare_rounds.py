import numpy as np
import pickle as pk
(test_counts,train_sections) = pk.load(open('categories.pkl','rb'))
(test_counts_r2,train_sections_r2) = pk.load(open('categories_round2.pkl','rb'))
print('computer missed, human detected')
r1 = np.array(test_counts['computer missed, human detected'])
r2 = train_sections_r2['computer missed, human detected']
idr1 = set([i[0] for i in r1])
idr2 = set([i[0] for i in r2])

is1_in2 = [i in idr2 for i in  idr1]
intersect_id = idr1.intersection(idr2)
intersect = r1[is1_in2]


from lib.sqlcontroller import SqlController
animal = 'DK55'

controller = SqlController(animal)
for celli in intersect:
    celli = celli[1]
    coord = np.array([celli['x'],celli['y'],celli['section']])*np.array([0.325,0.325,20])
    controller.add_layer_data_row(animal,34,1,coord,52,'shared misses between r1 and r2')