human(anastasia,woman, parent(null,null)).
human(ivan,man, parent(null,null)).
human(alexandra,woman, parent(null,null)).
human(yuri,man, parent(anastasia,null)).
human(zoya,woman, parent(null,null)).
human(viktor1, man, parent(alexandra,ivan)).
human(raisa, woman, parent(null,null)).
human(ludmila, woman, parent(zoya,yuri)).
human(alexandr, man, parent(raisa,viktor1)).
human(elena, woman, parent(raisa,viktor1)).
human(viktor2, man, parent(ludmila, alexandr)).
human(andrew, man, parent(elena, null)).
human(marina, woman, parent(null, null)).
human(konstantin, man, parent(marina, andrew)).


woman(X) :-
    human(X, woman, parent(_,_)).

man(X) :-
    human(X, man, parent(_,_)).

father(X,Y):-
    human(X, man, parent(_,_)), human(Y,_, parent(_, X)).

mother(X,Y):-
    human(X, woman, parent(_,_)), human(Y,_, parent(X,_)).

parent(X,Y):-
    father(X,Y); mother(X,Y).

son(X,Y):-
    (human(X, man, parent(_,Y)); human(X, man, parent(Y,_))),Y\=null.

daughter(X,Y):-
    (human(X, woman, parent(_,Y)); human(X, woman, parent(Y,_))),Y\=null.

brother(X,Y):-
    human(X, man, parent(_,Z)),human(Y, _, parent(_,Z)), X\=Y, Z\=null.
sister(X,Y):-
    human(X, woman, parent(_,Z)),human(Y, _, parent(_,Z)), X\=Y, Z\=null.

grandfather(X,Y):-
    human(X,man, parent(_,_)), human(Z, _, parent(_,X)), parent(Z,Y).

grandmother(X,Y):-
    human(X, woman, parent(_,_)), human(Z, _, parent(X,_)),  parent(Z,Y).

grandson(X,Y):- grandfather(Y,X).
granddaughter(X,Y):-grandmother(Y,X).
aunt(X,Y):-
   human(X, woman, parent(_,Z)),
   human(W, _, parent(_,Z)), W\=X, Z\=null,
    parent(W, Y).

uncle(X,Y):-
    human(X, man, parent(_,Z)), human(W, _, parent(_,Z)), W\=X, Z\=null,
    parent(W, Y).

niece(X,Y):-
    daughter(X,Z), (sister(Z, Y); brother(Z,Y)).

 nephew(X,Y):-
    son(X,Z), (sister(Z, Y); brother(Z,Y)).

?-son(ivan, X),write(X),nl.
