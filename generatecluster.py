# yohanes.gultom@gmail.com
# generate MEAD cluster from docsents
# usage: python generatecluster.py /home/yohanes/Workspace/duc/clean/06/D0608H/docsent > /home/yohanes/Workspace/duc/clean/06/D0608H/D0608H.cluster

import os
import sys

mead_home_dtd = '/home/yohanes/mead/dtd/cluster.dtd'

print '''<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE EXTRACT SYSTEM '{0}'>
<CLUSTER LANG='ENG'>'''.format(mead_home_dtd)
dir = sys.argv[1]
for f in os.listdir(dir):
    filepath = os.path.join(dir, f)
    if os.path.isfile(filepath):
        print '''<D DID='{0}' />'''.format(f.replace('.docsent', ''))
print '''</CLUSTER>'''
