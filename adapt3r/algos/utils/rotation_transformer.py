"""
Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/common/rotation_transformer.py
"""

from typing import Union, Literal, Callable
import adapt3r.utils.pytorch3d_transforms as pt
import torch
import numpy as np
import functools

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    # Dictionary mapping rotation representations to their sizes
    rep_sizes = {
        'matrix': 9,
        'quaternion': 4,
        'rotation_6d': 6,
        'axis_angle': 3,
        'euler_angles': 3
    }


    @classmethod
    def get_rep_size(cls, rep: str) -> int:
        """Get the size of a rotation representation.
        
        Args:
            rep: One of the valid rotation representations
            
        Returns:
            int: Size of the representation
        """
        assert rep in cls.valid_reps, f"Invalid representation {rep}"
        return cls.rep_sizes[rep]

    def __init__(
            self, 
            rep_in='axis_angle', 
            rep_network='rotation_6d',
            rep_out='matrix',
            convention_in=None,
            convention_network=None,
            convention_out=None):
        """
        Valid representations:
        - rep_in: Input representation
        - rep_network: Network representation (intermediate)
        - rep_out: Output representation

        Always uses matrix as intermediate representation for conversions.
        """

        self.rep_in = rep_in
        self.rep_network = rep_network
        self.rep_out = rep_out

        if rep_in == rep_network == rep_out:
            self.identity = True
        else:
            self.identity = False

        assert rep_in in self.valid_reps
        assert rep_network in self.valid_reps
        assert rep_out in self.valid_reps
        self.convention_in = convention_in
        self.convention_network = convention_network
        self.convention_out = convention_out

        if rep_in == 'euler_angles':
            assert convention_in is not None
    
        if rep_network == 'euler_angles':
            assert convention_network is not None
        if rep_out == 'euler_angles':
            assert convention_out is not None

    

    def convert(self, x: Union[np.ndarray, torch.Tensor], 
                from_rep: Union[Literal['input', 'network', 'output'], str], 
                to_rep: Union[Literal['input', 'network', 'output'], str]) -> Union[np.ndarray, torch.Tensor]:
        """Convert between any two representations.
        
        Args:
            x: Input rotation representation
            from_rep: Source representation ('input', 'network', 'output', or any valid representation)
            to_rep: Target representation ('input', 'network', 'output', or any valid representation)
            
        Returns:
            Converted rotation representation
        """
        if self.identity:
            return x
            
        # Map stage names to actual representations
        rep_map = {
            'input': self.rep_in,
            'network': self.rep_network,
            'output': self.rep_out
        }
        
        # Get actual representations, handling both stage names and direct representation names
        from_rep_actual = rep_map.get(from_rep, from_rep)
        to_rep_actual = rep_map.get(to_rep, to_rep)
        
        # Validate representations
        assert from_rep_actual in self.valid_reps, f"Invalid source representation: {from_rep}"
        assert to_rep_actual in self.valid_reps, f"Invalid target representation: {to_rep}"
        
        func = self._get_conversion_func(from_rep_actual, to_rep_actual)
        return self._apply_funcs(x, [func])

    def get_input_size(self) -> int:
        """Get the size of the input representation."""
        return self.get_rep_size(self.rep_in)

    def get_network_size(self) -> int:
        """Get the size of the network representation."""
        return self.get_rep_size(self.rep_network)

    def get_output_size(self) -> int:
        """Get the size of the output representation."""
        return self.get_rep_size(self.rep_out)

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y
    
    def preprocess(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        """Transform from input representation to network representation"""
        return self.convert(x, from_rep='input', to_rep='network')
    
    def postprocess(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        """Transform from network representation to output representation"""
        return self.convert(x, from_rep='network', to_rep='output')

    def _get_conversion_func(self, from_rep: str, to_rep: str):
        """Get the appropriate conversion function between two representations.
        
        Args:
            from_rep: Source representation
            to_rep: Target representation
            
        Returns:
            Callable: Function to convert between representations
        """
        if from_rep == to_rep:
            return lambda x: x
            
        if from_rep == 'matrix':
            func = getattr(pt, f'matrix_to_{to_rep}')
            if to_rep == 'euler_angles':
                convention = getattr(self, f'convention_{to_rep}')
                func = functools.partial(func, convention=convention)
        elif to_rep == 'matrix':
            func = getattr(pt, f'{from_rep}_to_matrix')
            if from_rep == 'euler_angles':
                convention = getattr(self, f'convention_{from_rep}')
                func = functools.partial(func, convention=convention)
        else:
            # Convert through matrix as intermediate
            func1 = getattr(pt, f'{from_rep}_to_matrix')
            func2 = getattr(pt, f'matrix_to_{to_rep}')
            if from_rep == 'euler_angles':
                convention = getattr(self, f'convention_{from_rep}')
                func1 = functools.partial(func1, convention=convention)
            if to_rep == 'euler_angles':
                convention = getattr(self, f'convention_{to_rep}')
                func2 = functools.partial(func2, convention=convention)
            func = lambda x: func2(func1(x))
            
        return func

    def __call__(self, *args, **kwargs):
        return self.preprocess(*args, **kwargs)

    def __repr__(self) -> str:
        """Returns a string representation of the RotationTransformer configuration.
        
        Returns:
            str: A string showing the input, network, and output representations
                 along with any conventions.
        """
        parts = [
            f"RotationTransformer(",
            f"    rep_in='{self.rep_in}'",
            f"    rep_network='{self.rep_network}'",
            f"    rep_out='{self.rep_out}'"
        ]
        
        if self.convention_in is not None:
            parts.append(f"    convention_in='{self.convention_in}'")
        if self.convention_network is not None:
            parts.append(f"    convention_network='{self.convention_network}'")
        if self.convention_out is not None:
            parts.append(f"    convention_out='{self.convention_out}'")
            
        parts.append(")")
        return "\n".join(parts)
    
    def __getattr__(self, name: str) -> Callable:
        """Dynamically handle conversion method names.
        
        Examples:
            - network_to_matrix: Convert from network representation to matrix
            - input_to_output: Convert from input representation to output representation
            - matrix_to_network: Convert from matrix to network representation
            
        Args:
            name: Method name in format 'from_to'
            
        Returns:
            Callable: Function that performs the requested conversion
        """
        if '_to_' not in name:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
        from_rep, to_rep = name.split('_to_')
        
        def conversion_func(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
            return self.convert(x, from_rep=from_rep, to_rep=to_rep)
            
        return conversion_func
    

def test():
    # Test in -> network -> out transformation
    tf = RotationTransformer(
        rep_in='axis_angle',
        rep_network='rotation_6d',
        rep_out='matrix'
    )

    rotvec = np.random.uniform(-2*np.pi, 2*np.pi, size=(1000,3))
    rot6d = tf.preprocess(rotvec)
    mat = tf.postprocess(rot6d)

    # Verify the transformation preserves rotation properties
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotation_6d will be normalized to rotation matrix
