function [H, A,b,Aeq, beq, lb, ub, opts] = LabelPropagationSettings(S,C,train_p_target,u,a)
[m, l] = size(train_p_target);
total = m*l;
b = sparse(total, 1);
%
D = diag(sum(S,2));
D2= diag(sum(C,2)); 
%%%%%%with no normalize
for zz=1:l
    %contruct the diagnal matrix of H firstly
    T=sparse(-1*u*S+(u+a)*eye(m)*(D(zz,zz)+D2(zz,zz)));
    %%%%
    H_row=T;
    for mm=1:zz-1
        %H_row=horzcat(sparse(eye(m)*D2(zz,mm)),H_row);
        %e=sparse(-1*eye(m)*C(zz,mm));
        H_row=horzcat(sparse(-1*a*eye(m)*C(zz,mm)),H_row);
    end
    for nn=zz:l-1
        %H_row=horzcat(H_row,sparse(eye(m)*D2(zz,nn+1)));
        %j=sparse(-1*eye(m)*C(zz,nn+1));
        H_row=horzcat(H_row,sparse(-1*a*eye(m)*C(zz,nn+1)));
    end
    if(zz==1)
         H=H_row;
    else
        H=vertcat(H,H_row);
    end
   H_row=sparse(m,m);
end
%%%%the original%%%%%%%%%%%%%%
% %Joint the whole matrix
% T=sparse(m,m);
% for ii = 1:m
%        for jj = 1:m
%             if(ii==jj)
%                 T(ii,jj) = 2*(u+1); 
%             else
%                 T(ii,jj) = -2*u*S(ii, jj)/(sqrt(D(ii,ii)*D(jj,jj)));
%             end
%        end
% end
% %T=2*(eye(m)*(u+1)-2*u*diag(power(diag(D),-1/2))*S*diag(power(diag(D),-1/2)));
% tep=T;
% for zz=1:l
%     H_row=T+sparse(a*eye(m)*D2(zz,zz));
%     for mm=1:zz-1
%         %H_row=horzcat(sparse(eye(m)*D2(zz,mm)),H_row);
%         H_row=horzcat(sparse(m,m),H_row);
%     end
%     for nn=zz:l-1
%         %H_row=horzcat(H_row,sparse(eye(m)*D2(zz,nn+1)));
%         H_row=horzcat(H_row,sparse(m,m));
%     end
%     if(zz==1)
%          H=H_row;
%     else
%         H=vertcat(H,H_row);
%     end
%    T=sparse(m,m);
%    H_row=sparse(m,m);
%    T=tep;
% end
%
% H2=sparse(total,total);
% for kk = 0:(l-1) 
%     for ii = 1:m
%        for jj = 1:m
%            if(ii == jj)%the diagnal elements
%                H2(ii+kk*m,jj+kk*m) = 2*(u+1)+D2(kk+1,kk+1); 
%            else 
%                H2(ii+kk*m,jj+kk*m) = -2*u*S(ii, jj)/(sqrt(D(ii,ii)*D(jj,jj))); % -2u*D^(-1/2)*S*D^(-1/2)
%            end
%            
%        end
%     end
% end
%
H=H+H';
A = sparse(total, total);
Aeq = sparse(m, total);
beq=ones(m,1);
for i = 1:m 
    Aeq_index=find(train_p_target(i,:)==1);
    %the number of labels
    beq(i)=length(Aeq_index);
    %the condition of ub
    train_p_target(i,Aeq_index)=length(Aeq_index);
    %
    Aeq_index=(Aeq_index-1)*m+i;
    Aeq(i,Aeq_index)=1;
    %A is diagol
    A_index=find(train_p_target(i,:)==1);
    A_index=(A_index-1)*m+i;
    for j = 1:length(A_index)
        A(A_index(j),A_index(j))=-1; 
    end
end
%case two:
% Aeq2 = sparse(m, total);
% beq2 =[ones(m,1);-1*ones(m,1)];
% for i = 1:m 
%     Aeq_index2=find(train_p_target(i,:)==-1);
%     Aeq_index2=(Aeq_index2-1)*m+i;
%     Aeq2(i,Aeq_index2)=1;
% end
% Aeq3=[Aeq;Aeq2];
%
%case one:
lb = -1*ones(total, 1);
ub = reshape(train_p_target, total, 1);
%
%case two:
% lb=reshape(train_p_target, total, 1);
% ub=reshape(train_p_target, total, 1);
% lb(logical(lb==1))=0;
% ub(logical(ub==-1))=0;
%
%case three:
% lb=sparse(total,1);
% ub=ones(total,1);
% for i = 1:m 
%    Aeq(i, i:m:total) = 1;
% end
%
opts = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','Display','off');
end