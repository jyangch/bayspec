"""Parameter and configuration objects used to build models."""

from .post import Post
from .prior import Prior



class Par(object):
    """Model parameter carrying value, prior, posterior, and metadata.

    ``Par`` instances can be linked to share value, prior, and posterior
    across several models; setting any of those on one mate propagates the
    change to the others. A version counter is bumped on every value
    change so downstream caches can invalidate.

    Attributes:
        val: Current numerical value.
        prior: Optional prior distribution.
        post: Optional posterior sample container.
        comment: Free-form description.
        scale: ``'linear'`` or ``'log'`` — determines how ``value`` is built.
        unit: Optional unit attached to ``val``.
        frozen: If ``True``, the parameter is held fixed during inference.
        mates: Set of other ``Par`` instances kept in sync via ``link``.
    """

    def __init__(self,
                 val,
                 prior=None,
                 post=None,
                 comment=None,
                 scale='linear',
                 unit=None,
                 frozen=False
                 ):
        """Create a parameter with the given value and metadata.

        Args:
            val: Initial numerical value.
            prior: Optional prior distribution.
            post: Optional posterior sample container.
            comment: Free-form description.
            scale: ``'linear'`` or ``'log'``.
            unit: Optional unit attached to ``val``.
            frozen: If ``True``, the parameter is fixed during inference.
        """

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
        
        return self._val


    @val.setter
    def val(self, new_val):
        """Set ``val`` and propagate it to every linked mate; bumps ``version``."""

        self._val = new_val

        for mate in self.mates:
            if mate.val != self.val:
                mate.val = self.val

        self._bump_version()


    def _bump_version(self):
        
        self._version += 1
        

    @property
    def version(self):
        """Monotonic counter bumped on every value change.

        Downstream code uses this as a cheap cache key.
        """

        return self._version
                
                
    @property
    def prior(self):

        return self._prior
        
        
    @prior.setter
    def prior(self, new_prior):
        """Set ``prior`` and propagate it to every linked mate.

        Raises:
            ValueError: If ``new_prior`` is neither a ``Prior`` nor ``None``.
        """

        if isinstance(new_prior, (Prior, type(None))):
            self._prior = new_prior
        else:
            raise ValueError('Unsupported prior type')

        for mate in self.mates:
            if mate.prior != self.prior:
                mate.prior = self.prior


    @property
    def prior_info(self):
        """Short description of the prior, or ``None`` when unset."""

        if isinstance(self.prior, Prior):
            return self.prior.info
        else:
            return None


    @property
    def post(self):
        
        return self._post
    
    
    @post.setter
    def post(self, new_post):
        """Set ``post`` and propagate it to every linked mate.

        Raises:
            ValueError: If ``new_post`` is neither a ``Post`` nor ``None``.
        """

        if isinstance(new_post, (Post, type(None))):
            self._post = new_post
        else:
            raise ValueError('Unsupported post type')

        for mate in self.mates:
            if mate.post != self.post:
                mate.post = self.post
                
                
    @property
    def post_info(self):
        """Summary of the posterior, or ``None`` when unset."""

        if isinstance(self.post, Post):
            return self.post.info
        else:
            return None


    def link(self, other):
        """Link this parameter with ``other`` to keep them synchronized.

        After linking, setting ``val``/``prior``/``post`` on either side
        propagates to every mate in the group.

        Args:
            other: Another ``Par`` instance to link with.

        Raises:
            ValueError: If ``other`` is the same instance as ``self``.
        """

        assert isinstance(other, Par)

        if id(self) == id(other):
            raise ValueError('cannot link itself')

        self.mates.add(other)
        other.mates.add(self)

        self.val = self._val
        self.prior = self._prior
        self.post = self._post


    def unlink(self, other):
        """Break the link between this parameter and ``other``."""

        assert isinstance(other, Par)

        self.mates.discard(other)
        other.mates.discard(self)


    def frozen_at(self, new_val):
        """Set the value to ``new_val`` and freeze the parameter."""

        self.val = new_val
        self.frozen = True


    @property
    def value(self):
        """Physical value derived from ``val``, ``scale``, and ``unit``.

        For ``scale='linear'`` this returns ``val`` (optionally times the
        unit); for ``scale='log'`` it returns ``10 ** val``.

        Raises:
            ValueError: If ``scale`` is neither ``'linear'`` nor ``'log'``.
        """

        if self.scale == 'linear':
            if self.unit is None:
                return self.val
            else:
                return self.val * self.unit
        elif self.scale == 'log':
            if self.unit is None:
                return 10 ** self.val
            else:
                return 10 ** self.val * self.unit
        else:
            raise ValueError('invalid parameter scale')


    @property
    def range(self):
        """Plausible range for the parameter.

        Returns a collapsed point if the parameter is frozen, the full
        uniform support for uniform priors, or the central 95% interval
        otherwise.
        """

        if self.frozen:
            return (self.val, ) * 2
        elif self.prior.expr == 'unif':
            return self.prior.interval(1.0)
        else:
            return self.prior.interval(0.95)


    def todict(self):
        """Serialize the parameter into a plain-dict representation."""

        return {'val': self.val,
                'prior': self.prior_info,
                'post': self.post,
                'comment': self.comment,
                'scale': self.scale,
                'unit': self.unit,
                'frozen': self.frozen}


    @property
    def info(self):
        """Print each field of ``todict`` one per line and return an empty string.

        The empty return lets ``info`` be used transparently as a property
        in an interactive session where the print output is the point.
        """

        for key, value in self.todict().items():
            print(f'> {key}: {value}')

        return ''



class Cfg(Par):
    """Configuration parameter: a ``Par`` permanently frozen with no prior.

    ``Cfg`` represents values that configure a model (e.g. sample redshift
    or selected channel) rather than quantities that inference should
    estimate. They accept the same constructor signature as ``Par`` but
    ``prior`` and ``post`` are forced to ``None`` and ``frozen`` to
    ``True``.
    """

    def __init__(self,
                 val,
                 prior=None,
                 post=None,
                 comment=None,
                 scale='linear',
                 unit=None,
                 frozen=False
                 ):
        """Create a configuration parameter.

        Args:
            val: Configuration value.
            prior: Ignored; always forced to ``None``.
            post: Ignored; always forced to ``None``.
            comment: Free-form description.
            scale: ``'linear'`` or ``'log'``.
            unit: Optional unit.
            frozen: Ignored; always forced to ``True``.
        """

        super().__init__(val, prior, post, comment, scale, unit, frozen)

        self.prior = None
        self.post = None
        self.frozen = True
