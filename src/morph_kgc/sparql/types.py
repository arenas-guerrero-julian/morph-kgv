__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero"]

__license__ = "Apache-2.0"
__maintainer__ = "Julián Arenas-Guerrero"
__email__ = "julian.arenas.guerrero@upm.es"


from typing import Optional, Set, Tuple, Union
from rdflib import Literal, URIRef, Variable

Term = Union[URIRef, Literal]
Triple = Tuple[Term, Term, Term]
TriplePattern = Union[URIRef, Literal, Variable]
SearchQuery = Tuple[Optional[Term], Optional[Term], Optional[Term]]
BGP = Set[TriplePattern]