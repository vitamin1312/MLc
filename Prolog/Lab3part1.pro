man(ivan).
man(yuri).
man(viktor1).
man(alexandr).
man(viktor2).
man(andrew).
man(konstantin).

wonam(anastasia).
woman(alexandra).
woman(zoya).
woman(raisa).
wonan(ludmila).
woman(elena).
woman(marina).

parent(anastasia,yuri).
parent(ivan,viktor1).
parent(alexandra,viktor1).
parent(yuri,ludmila).
parent(zoya,ludmila).
parent(viktor1,alexandr).
parent(raisa,alexandr).
parent(viktor1,elena).
parent(raisa,elena).
parent(ludmila,viktor2).
parent(alexandr,viktor2).
parent(elena,andrew).
parent(andrew,konstantin).
parent(marina,konstantin).


child(X,Y):- parent(Y,X).

wife(X,Y):-
  child(Z,X), child(Z,Y), woman(X).

husband(X,Y):- wife(Y,X).

son(X,Y):-
  parent(Y,X),man(X).

daughter(X,Y):-
  parent(Y,X),woman(X).

sister(X,Y):-
  parent(Z,X), parent(Z,Y), woman(X), X\=Y.

brother(X,Y):-
  parent(Z,X), parent(Z,Y), man(X), X\=Y.

father(X,Y):-
  parent(X,Y),man(X).

mother(X,Y):-
  parent(X,Y),woman(X).

grandfather(X, Y):-
  parent(Parent, Y),
  father(X, Parent).

grandmother(X, Y):-
  parent(Parent, Y),
  mother(X, Parent).

aunt(X,Y) :-
  (   parent(Z,Y),
  sister(X,Z));(husband(Z,X),parent(E,Y),brother(Z,E)).

uncle(X,Y) :-
  (   parent(Z,Y),
  brother(X,Z));(husband(Z,X),parent(E,Y),brother(Z,E)).


niece(X,Y):-
 daughter(X,Z),
 brother(Z,Y);
 daughter(X,Z),
 sister(Z,Y).


?-uncle(elena,X),write(X),nl.
