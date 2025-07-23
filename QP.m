function [P,fval] = LabelPropagation(Y, H, A,b,Aeq, beq, lb, ub, opts)
[m, l] = size(Y);
y = reshape(Y, m*l, 1);
f = -2*y;
[p,fval] = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], opts);
P = reshape(p, m, l);
end