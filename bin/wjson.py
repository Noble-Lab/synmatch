import sys
import numpy as np
import json
import os


def makeJson(n1, n2, limit=1, name="my intersection", name_left="matroid 1", name_right="matroid 2", comment="my comment"):
    "Produces a dictionary in the coopraize format. N is the number of cells."
      
    left = [np.array(range(0,n2)) + np.repeat(n2,n2)*i for i in range(n1)]
    right = [np.repeat(n2,n1)*np.array(range(n1))+i*np.repeat(1,n1) for i in range(n2)]
    
    json_object = {
        "intersection-of-matroids" :
        {
        "name" : name,
        "comment": comment,
        "partition-matroids" :
            [{
                "partition-matroid": {
                    "name": "matroid 1",
                    "blocks": [{"block":[int(v) for v in block], "limit":int(limit)} for block in left]
                }
            },
            {
                "partition-matroid": {
                    "name": "matroid 2",
                    "blocks": [{"block":[int(v) for v in block], "limit":int(limit)} for block in right]
                }
            }]
        }
    }
    return(json_object)


def writeJson(ncells1, ncells2, jname):
    if os.path.exists(jname): return

    json_object = makeJson(ncells1, ncells2)
	
    file = open(jname, "w")
    file.write(json.dumps(json_object, indent=4))
    file.close()

