
# Mapping of aggregate to specific
di_agg = {'nsi1':['othsysep','othseshock'], # non-site infection
          'nsi2':['othsysep','othseshock','urninfec'],
          'nsi3':['othsysep','othseshock','urninfec','othclab'],
          'nsi4':['oupneumo','urninfec','othclab','othsysep','othseshock'],
          'ssi1':['supinfec','wndinfd','orgspcssi'],
          'ssi2':['supinfec','wndinfd','orgspcssi','sdehis','dehis'],
          'aki':['renainsf', 'oprenafl'],
          'adv1':['death30yn','renainsf', 'oprenafl','cdarrest'],
          'adv2':['death30yn','reintub', 'reoperation','cdarrest','cnscva'],
          'unplan1':['readmission1','reoperation'],
          'unplan2':['readmission1','reoperation','reintub'],
          'cns':['cnscva','cszre','civhg']}


# Clean labels
di_outcome = {'adv':'ADV', 'aki':'AKI', 'cns':'CNS',
              'nsi':'nSSIs', 'ssi':'SSIs', 'unplan':'UPLN'}