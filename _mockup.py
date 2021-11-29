import string
import pandas as pd
import numpy as np
#see dir(example_state).log, lauter unabhängige seeds für reproducibility
local_state = lambda seed: np.random.RandomState(seed)

ALL = string.ascii_letters
UPPER = string.ascii_uppercase
LOWER = string.ascii_lowercase
DIGITS = string.digits

def test_prime(n:int)->bool:
    i=1
    while i**2<=n:
        i+=1
        if n%i==0:
            return False
    return True

def get_generators_for_prime(p:int):
    #assert test_prime(p), "this only works for prime p, sorry"
    d = {g:[g**i % p for i in range(1,p)] for g in range(2,p)}
    M = max([len(d[g]) for g in d.keys()])
    d_gen = {g:d[g] for g in d.keys() if len(set(d[g]))==M}
    return d_gen

class NoiseSource():
    def __init__(self, seed:int=42, MOCKUP_SIZE:int=240000, VALID_CHARACTERS=ALL+DIGITS+'=-!?()+'):
        def noisy_keygen(MOCKUP_SIZE:int=240000, seed:int=42, VALID_CHARACTERS=ALL+DIGITS+'=-!?()+'):
            def mult(l:list):
                k=1                
                for i in list(l):
                    k*=i
                return k
            def strtoint():        
                from collections import defaultdict
                rng=local_state(seed).randint
                strtoint={char:rng(22222,33333)*97 for char in VALID_CHARACTERS}
                del rng
                return lambda x: strtoint[x]
            def braidy_sort_key(key:str=''):
                key = str(key)
                if len(key)<=18:
                    return sorted([key[:6],key[6:]],key=lambda x: (len(x),x))
                if len(key)>=7+12*2:
                    a,b=sorted((key[7:11],key[15:19]))
                    return [key[:7], b, '4'+key[12:15], key[11:12]+a[1:], 
                            a[0]+''.join(sorted(list(key[19:26]))),key[26:]]
                return sorted([key[:3],key[3:6],key[6:9],key[9:]],key=lambda x: (len(x),x))    
            key_size=120
            import time
            strtoint=strtoint()
            ch = local_state(seed).choice
            
            KEY_TEXT_SOURCE=''.join(ch(list(VALID_CHARACTERS)) for _ in range(MOCKUP_SIZE*key_size))
            
            mini_ps = [3,5,7,13,11,2,17,19,21,23,11,2,17,19,21,23]
            ps = [29,31,37,41,43,67,71,73,97,89,43,67,71,73,97,89]                  
            r = [739,743,751,757,761,769,773,787,797,809,811,769,773,787,797,809,811]
            u = lambda z: sum(map(lambda x: x[0]%x[1], zip(z,r)))
            v = lambda z: mult(map(lambda x: x[0]%x[1], zip(z,ps)))
            w = lambda z: mult(map(lambda x: (x[0]**x[1])%x[2], zip(z,ps,r)))            
            isn = lambda z: sorted(list(map(lambda x: str(((x[0]**x[1])%x[2]+1297)**3)[3:8], zip(z,ps,r))))
            jsn = lambda z: sorted(list(map(lambda x: str(((x[0]**x[1])%x[2]+1201)**3)[1:7], zip(z,ps,r))))
            step = 0
            while len(KEY_TEXT_SOURCE) >= key_size:        
                step += 1                                
                key, KEY_TEXT_SOURCE = KEY_TEXT_SOURCE[:key_size], KEY_TEXT_SOURCE[key_size:]
                tmp = sorted(list(map(strtoint,list(key))))
                bskey = braidy_sort_key(key)
                #sum(map(lambda x: mult(map(lambda ix: (x**ix[0]) % ix[1] , enumerate(ps))), tmp)),
                #sum(map(lambda x: x**2 % 4514113 , tmp)),                      
                #sum(map(lambda x: x**3 % 4690451 , tmp)),
                #sum(map(lambda x: (13*x)**3 % 9843019, tmp)),
                #sum(map(lambda x: x**4 % 467, tmp)),                      
                tmp = [time.time(),(time.time() % 17)**2, time.time() % 313 ]+['-'.join(bskey[:-1]),] + [bskey[-1],] + \
                        [''.join(ch(list(LOWER+'-')) for _ in range(40)),] +\
                       isn(tmp) + jsn(tmp) + [u(tmp) , v(tmp) , w(tmp)] +\
                      sorted([\
              ''.join(ch(list('aeuaeaeaaaeuooomnpklbbddststst'+'-----')) for _ in range(20)),
              ''.join(ch(list('aeuaeaeaaaeuooomnpklqpxyz'+'-----')) for _ in range(20)),
              ''.join(ch(list('aeuaeaeaaaeuoootttrrrrrddststst'+'-----')) for _ in range(20)),
              ''.join(ch(list('aeuaeaeaaaeuooomllllldststst'+'-----')) for _ in range(20)),                      
                      ])+\
                      sorted([#sum(tmp)**3,                      
                      sum(map(lambda x: (x**3 % 313)*(x**7 % 379)*(x**10 % 467), tmp)),                       
                      sum(map(lambda x: int(np.sin(x//97)*1000) % 467, tmp)),                      
                      sum(map(lambda x: int(np.log1p(np.abs(x))) % 467, tmp)),
                      sum(map(lambda x: x**2 % 467, tmp)),
                      sum(map(lambda x: (x//100)**2 % 16, tmp)),
                      sum(map(lambda x: (x%10)**2 % 16, tmp)),
                      sum(map(lambda x: (x%5)**2 % 7, tmp))%13,
                      sum(map(lambda x: (x%7)**2 % 16, tmp)),
                      sum(map(lambda x: (x%7)**2 % 16, tmp))%9,                      
                      sum(map(lambda x: (13*x)**3 % 4514113, tmp)),                                                                  
                      sum(map(lambda x: x**2 % 9843019, tmp)),                      
                      sum(map(lambda x: (13*x)**4 % 4695947 , tmp)),] ,
                            key=lambda x: (len(str(x)),str(x)) )                
                if step%100 == 0:
                    print(tmp)                                            
                yield tmp
        #__init__ ctd.
        import pandas as pd
        self.ng = noisy_keygen(MOCKUP_SIZE=MOCKUP_SIZE,seed=seed,VALID_CHARACTERS=VALID_CHARACTERS)
        
    def make_noisy_csv(self,fname=''):
        import pandas as pd
        print('reading into dataframe, first ng-call')
        self.df = pd.DataFrame(self.ng)
        if len(str(fname))==0:
            self.df.to_csv('noise_source.csv')
        self.df.to_csv(str(fname)+'.csv')

for i in range(100,390,13):
    import time
    ns = NoiseSource(seed=i,MOCKUP_SIZE=22048)
    ns.make_noisy_csv(str(int(time.time()))+'_'+str(22048)+'-'+str(i))
