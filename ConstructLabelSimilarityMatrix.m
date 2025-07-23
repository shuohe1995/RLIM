function [C] = ConstructLabelSimilarityMatrix( train_p_target )
%calculate label similarity with cosine similarity.
    Y=train_p_target;
    C=1-pdist2(Y',Y','cosine');
    C=(C+1)/2;
    C=C-diag(diag(C));   
end

