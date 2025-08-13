#DEPRECATED - DELETE
#QUESTIONS: SEE DUANE
#11-AUG-2025

import numpy as np
class CellAnnotationUtilities:
    #TODO: type in name; potentially colsolidate this class with something else?
    def find_cloest_neighbor_among_points(self,all_points,search_array,max_distance = 20,verbose_unequal = False,verbose_skipping = True):
        index_array = []
        i = 0
        for celli in search_array:
            if len(celli)==3:
                section = celli[2]
                in_section = all_points[:,2]==section
                if np.any(in_section) == False:
                    continue
                segments_in_section = all_points[in_section,:2]
                diff = segments_in_section[:,:2]-celli[:2]
            elif len(celli)==2:
                diff = all_points - celli
            dist = np.sqrt(np.sum(np.square(diff),axis=1))
            cloest_segment = np.argmin(dist)
            if len(celli)==3:
                corresponding_id = np.where(np.cumsum(in_section)==cloest_segment+1)[0][0]
            else:
                corresponding_id = cloest_segment
            if dist[cloest_segment]==0:
                index_array.append(corresponding_id)
            elif dist[cloest_segment]<max_distance:
                index_array.append(corresponding_id)
                if verbose_unequal:
                    print(f'cannot find equal,subbing point with distance: {dist[cloest_segment]}')
            else:
                if verbose_skipping:
                    print(f'skipping, min distance {dist[cloest_segment]}')
                continue
            if i%1000 == 0 and i !=0:
                print(i)
            i+=1
        return index_array