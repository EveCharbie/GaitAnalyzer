function result = fillgap(MATRICE)
% ---------------------------------
% FILLGAP: restaure les donn�es manquantes des trajectoires fournies par
% Vicon, elles se traduisent toujours par des z�ros
% - l'algorithme interpole entre tous les z�ros des donn�es fournies
% - pour faire face aux z�ros aux bords, une proc�dure d'augmentation des donn�es
%   (identique � celle pour utilis�e pour le filtrage) est utilis�e


% INPUT:
%   MATRICE 	: temps x Nbre_cannaux
% OUTPUT
%   result 		: matrice de m�me dimension que MATRICE mais 'n�ttoy�e'
% ----------------------------------
%%

% initialisations
result  = MATRICE;
[nl,nc] = size(MATRICE);

% op�ration pour chaque colonne
for i = 1:nc
    
    if sum(MATRICE(:,i) == 0) == 0
        result(:,i) = MATRICE(:,i);
        
    elseif sum(MATRICE(:,i) == 0) == nl
        %fprintf('FILLGAP: attention: nullit� de la colonne %u\n',num2str(i));
        % do nothing
    else
        % v�rifier la non existence de zeros aux bords
        % et utiliser la duplication des donn�es si vrai
        if MATRICE(1,i)==0 || MATRICE(end,i)==0
            try
                [temp,N] = parTrois(MATRICE(:,i));
                temp = interpoler(parTrois(temp));
                result(:,i) = temp(N:end-N);
            catch
               %disp(e.message);
               %fprintf('FILLGAP: impossible de r�cup�rer la colonne %u\n',i);
               result(:,i) = ones(nl,1)*mean(MATRICE(MATRICE(:,i)~=0,i));
            end
            
        else
            result(:,i) = interpoler(MATRICE(:,i));
            
        end
    end
end

%% SOUS FONCTIONS
    function [y,nbre_pts] = parTrois(X,opt)
        % m�thode d'Amarantini pour �viter les hyst�r�ses aux bornes
        % INPUT:
        %  X est le vecteur � �tendre
        %  opt = est le pourcentage de donn�es � utiliser, par d�faut = 0.2
        % OUTPUT
        % y : le vecteur �tendu
        % nbre_pts : les nombre de points suppl�mentaire aux deux bords

        % initialisation
        if nargin <2
            opt =0.2;
        end
        X = X(:);
        N_ = length(X);
        nbre_pts = floor(opt*N_);
       
        % concat�nation
        origine1 = mean(X(1:floor(opt*N_/5)));
        origine2 = mean(X(end-floor(opt*N_/5):end));
       
        y = [-flipud(X(1:nbre_pts))+2*origine1;X(2:end-1);-flipud(X(end-nbre_pts:end))+2*origine2]';
    end
    
    % fonction d'interpolation par cubic sline
    function result = interpoler(X)
        x_ref_  = 1:length(X);
        indice  = find(X ~= 0);
        result  = interp1(x_ref_(indice)',X(indice),x_ref_,'splin');
    end



end % end of all