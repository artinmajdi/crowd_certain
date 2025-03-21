from crowd_certain.utilities.parameters.params import ConsistencyTechniques, ConsistencyTechniqueType
from typing import List, Type, TypeVar, Union, Literal, TypeAlias, get_args

# Existing type variable (if needed in other parts of your code)
ConsistencyTechniquesType2 = TypeVar( 'ConsistencyTechniquesType2', bound=ConsistencyTechniques )

# Type alias for the ConsistencyTechniques class itself
ConsistencyTechniquesClass: TypeAlias = Type[ConsistencyTechniques]

# Example usage: specifying a parameter or return type as the ConsistencyTechniques class
def get_technique_class() -> ConsistencyTechniquesClass:
    return ConsistencyTechniques

# You can still call class methods like values()
print(ConsistencyTechniques.values())

# ConsistencyTechniqueType2: TypeAlias = Literal[ *ConsistencyTechniques.values() ]

print(get_args(ConsistencyTechniquesType2))
print('this')
