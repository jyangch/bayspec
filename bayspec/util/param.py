from .post import Post
from .prior import Prior



class Par(object):

    def __init__(self, 
                 val, 
                 prior=None, 
                 post=None, 
                 comment=None, 
                 scale='linear', 
                 unit=1, 
                 frozen=False
                 ):
        
        self._val = val
        self._prior = prior
        self._post = post
        self._version = 0
        
        self.comment = comment
        self.scale = scale
        self.unit = unit
        self.frozen = frozen
        self.mates = set()


    @property
    def val(self):
        
        return float(self._val)


    @val.setter
    def val(self, new_val):
        
        self._val = new_val
        
        for mate in self.mates:
            if mate.val != self.val:
                mate.val = self.val

        self._bump_version()


    def _bump_version(self):
        
        self._version += 1
        

    @property
    def version(self):
        
        return self._version
                
                
    @property
    def prior(self):

        return self._prior
        
        
    @prior.setter
    def prior(self, new_prior):
        
        if isinstance(new_prior, (Prior, type(None))):
            self._prior = new_prior
        else:
            raise ValueError('Unsupported prior type')
        
        for mate in self.mates:
            if mate.prior != self.prior:
                mate.prior = self.prior


    @property
    def prior_info(self):
        
        if isinstance(self.prior, Prior):
            return self.prior.info
        else:
            return None


    @property
    def post(self):
        
        return self._post
    
    
    @post.setter
    def post(self, new_post):
        
        if isinstance(new_post, (Post, type(None))):
            self._post = new_post
        else:
            raise ValueError('Unsupported post type')

        for mate in self.mates:
            if mate.post != self.post:
                mate.post = self.post
                
                
    @property
    def post_info(self):
        
        if isinstance(self.post, Post):
            return self.post.info
        else:
            return None


    def link(self, other):
        
        assert isinstance(other, Par)
        
        if id(self) == id(other):
            raise ValueError('cannot link itself')
        
        self.mates.add(other)
        other.mates.add(self)
        
        self.val = self._val
        self.prior = self._prior
        self.post = self._post
        
        
    def unlink(self, other):
        
        assert isinstance(other, Par)
        
        self.mates.discard(other)
        other.mates.discard(self)


    def frozen_at(self, new_val):
        
        self.val = new_val
        self.frozen = True
        
        
    @property
    def value(self):
        
        if self.scale == 'linear':
            return self.val * self.unit
        elif self.scale == 'log':
            return 10 ** self.val * self.unit
        else:
            raise ValueError('invalid parameter scale')
    
    
    @property
    def range(self, q=0.95):
        
        if self.frozen:
            return (self.val, ) * 2
        else:
            return self.prior.interval(q)


    def todict(self):
        
        return {'val': self.val, 
                'prior': self.prior_info, 
                'post': self.post, 
                'comment': self.comment, 
                'scale': self.scale, 
                'unit': self.unit, 
                'frozen': self.frozen}


    @property
    def info(self):
        
        for key, value in self.todict().items():
            print(f'> {key}: {value}')
            
        return ''



class Cfg(Par):
    
    def __init__(self, 
                 val, 
                 prior=None, 
                 post=None, 
                 comment=None, 
                 scale='linear', 
                 unit=1, 
                 frozen=False
                 ):
        
        super().__init__(val, prior, post, comment, scale, unit, frozen)
        
        self.prior = None
        self.post = None
        self.frozen = True
