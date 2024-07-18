import sys

import os

tasks = ['longbook_choice_eng', 'longbook_qa_eng']
#toplist = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
toplist = [5, 10, 15] #20, 30, 40, 50, 60, 70, 80, 90, 100]

types = {
    'txt_0920.202407' : 'full',
    'txt_0920.cut.202407' : 'cut',
    'txt_0920.same.202407' : 'same',
}

def find_task(afn):
    for task in tasks:
        if task in afn:
            return task
    return 'NA'

def find_topn(afn):
    for atop in toplist:
        atop = str(atop)
        atopstr = '_{}_generate'.format(atop)
        if atopstr in afn:
            return atop
    return '-1'

def type3(afn):
    for atype in types:
        if atype in afn:
            return types[atype]
    return 'NA'

# task : 

#         task full same cut
# top-5
# top-10

#for aline in sys.stdin:
#for aline in open('1and2.comb_comscore.tstout.ref.0709.sh.log.v2'):
skeys = ['74', '296', '370', '444', '592', '368']

for abigkey in skeys:
    abigkeyfull = '_{}.txt'.format(abigkey)
    abigkeyret = "_{}_ret.txt".format(abigkey)

    for asmallkey in [abigkeyfull, abigkeyret]:
        print(asmallkey)
        
        out_dict = dict()
        #import ipdb; ipdb.set_trace()
        #for aline in open('1and2.comb_comscore.tstout.ref.0711.longbook.sh.log'):
        for aline in open('1and2.comb_comscore.tstout.ref.0715.longbook.sh.log.v2'):
            if 'final display: ' in aline and asmallkey in aline:
                aline = aline.strip()
                cols = aline.split(' ')
                if len(cols) < 3:
                    continue


                atask = find_task(aline)
                if atask == 'longbook_choice_eng' and aline.endswith('False'):
                    continue
            
                atop = find_topn(aline)
                atype = type3(aline)
                #import ipdb; ipdb.set_trace()

                #print(cols[:5])
                #for i in range(5):
                #    print(i, cols[i])

                ascore = cols[3]
                #print(atask, atop, atype, ascore)

                # 0-th outdict = {task : topdict}
                topdict = out_dict[atask] if atask in out_dict else dict()

                # 1-th topdict = {top : typedict}
                typedict = topdict[atop] if atop in topdict else dict()

                # 2-th typedict = {type : score}
                typedict[atype] = ascore

                topdict[atop] = typedict

                out_dict[atask] = topdict

                #print(out_dict)


        #print(out_dict)

        for atask in out_dict:
            topdict = out_dict[atask]

            for atop in toplist:
                atop = str(atop)
                if atop in topdict:
                    typedict = topdict[atop]
                    allscores = ''
                    for atype in ['full', 'same', 'cut']:
                        ascore = typedict[atype]
                        allscores += ' ' + ascore
                    print('{} {} {} {} {}'.format(atask, atop, allscores, asmallkey, abigkey))

                    #for atype in typedict:
                    #    ascore = typedict[atype]
                    #    print('{}\t{}\t{}\t{}'.format(atask, atop, atype, ascore))








