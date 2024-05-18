human(anastasia,woman, parent(null,null),1910).
human(ivan,man, parent(null,null),1912).
human(alexandra,woman, parent(null,null),1916).
human(yuri,man, parent(anastasia,null),1947).
human(zoya,woman, parent(null,null),1946).
human(viktor1, man, parent(alexandra,ivan),1940).
human(raisa, woman, parent(null,null),1949).
human(ludmila, woman, parent(zoya,yuri),1975).
human(alexandr, man, parent(raisa,viktor1),1976).
human(elena, woman, parent(raisa,viktor1),1973).
human(viktor2, man, parent(ludmila, alexandr),2002).
human(andrew, man, parent(elena, null),1992).
human(marina, woman, parent(null, null),1991).
human(konstantin, man, parent(marina, andrew),2020).


husband(X,Y):-
    human(_,_,parent(X,Y),_),X\=null,Y\=null.
wife(Y,X):- husband(X,Y).
    %human(_,_,parent(X,Y),_),X\=null,Y\=null.
son(X,Y):-
    (human(X,man,parent(Y,_),_);human(X,man,parent(_,Y),_)),Y\=null.
daughter(X,Y):-
    (human(X,woman,parent(Y,_),_);human(X,woman,parent(_,Y),_)),Y\=null.

mother(X,Y):-
    human(Y,_,parent(X,_),_),X\=null.
father(X,Y):-
    human(Y,_,parent(_,X),_),X\=null.

brother(X,Y):-
    human(X,man,parent(Z,V),_),human(Y,_,parent(Z,V),_),(Z\=null;V\=null),X\=Y.
sister(X,Y):-
    human(X,woman,parent(Z,V),_),human(Y,_,parent(Z,V),_),(Z\=null;V\=null),X\=Y.

granddad(X,Y):-
    human(Z,_,parent(_,X),_),(human(Y,_,parent(Z,_),_);human(Y,_,parent(_,Z),_)),X\=null.
grandmom(X,Y):-
    human(Z,_,parent(X,_),_),(human(Y,_,parent(Z,_),_);human(Y,_,parent(_,Z),_)),X\=null.

aunt(X,Y):-
    (sister(X,Z);sister(X,V)),human(Y,_,parent(Z,V),_),Z\=null,V\=null.
uncle(X,Y):-
    (brother(X,Z),Z\=null,human(Y,_,parent(Z,_),_); brother(X,V),V\=null,human(Y,_,parent(_,V),_)).
age(X,Y):-
    human(X,_, parent(_,_),Z),Y is (2023-Z),Z\=0.

child(X,Y):-
    (son(X,Y);daughter(X,Y)).
list_length([],0).
list_length([0|T1],N):-
   length(T1,N1),N=N1+1.

list_of_children(X,List):-
    findall(C,child(C,X),List).

num_of_children(X,N):-
    list_of_children(X,List),
    list_length(List,N).


workers(X):-human(X,_,parent(_,_),Y),Y<2005,Y>1958.
list_of_workers(List):-findall(X,workers(X),List).

pensioners(X):-human(X,_,parent(_,_),Y),Y=<1958.
list_of_pensioners(List):-findall(X,pensioners(X),List).


?-aunt(elena, X),write(X),nl.

