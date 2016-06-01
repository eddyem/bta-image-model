
function make_mask()
% ���������� ������������� �����
% SS - ������ �����
	f = fopen("holes.json", "w");
	X = []; Y = [];
	R = [175 247 295 340 379 414 448 478] * 1e-3; % ������� ����� �� ��������������
	HoleR = 7.5e-3; % ������ ��������� - 7.5��
	R0 = .6;  % ������ ����� ��������������
	alpha0 = pi/32; % �������� ����� ������������ ������� �����������/���������
	Angles = [0:31] * 2 * alpha0 + alpha0; % ����, �� ������� ������������� ����
	% ��� ����, ����� ���������� �� ����� ����������, �������� �����
	% ����������: zeros(15) � ��������� ���, ��� ������ ���� �����. �����
	% ������� ��������� � mask �� �����, ���� ������ �������� ����� �������
	fprintf(f, "{\n\t\"maskz\": 20.017,\n\t\"shape\": \"round\", \"radius\": %f,\n\t\"holes\": [\n" , HoleR);
	for i = [1 : size(R,2)] % ���� �� �������
		x = R(i) * cos(Angles);
		y = R(i) * sin(Angles);
		X = [X x]; Y = [Y y];
		%fprintf(f, "\t\t{\"ring\": %d, \"center\": [%f, %f]\n", i, x[j], y[j]]);
		printR(f, sprintf("\"ring\": %d", i-1), x, y);
	endfor
	% �������� �������
	x = R([8 3]) .* cos([-2 -7]* 2 * alpha0);
	y = R([8 3]) .* sin([-2 -7]* 2 * alpha0);
	X = [X x]; Y = [Y y];
	%fprintf(f, "\t\t{\"marker\", \"center\": [%f, %f]\n", x, y);
	printR(f, sprintf("\"mark\": 1"), x, y);
	fprintf(f, "\t]\n}\n");
	fclose(f);
	plot(X, Y, 'o'); axis square;
endfunction

function printR(f, msg, x, y)
	for i = 1:size(x,2)
		fprintf(f, "\t\t{ %9s, \"number\": %2d, \"center\": [ %7.4f, %7.4f ] },\n", msg, i-1, x(i), y(i));
	endfor
endfunction
