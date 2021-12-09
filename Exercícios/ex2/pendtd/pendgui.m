%#ok<*DEFNU>

function varargout = pendgui(varargin)
% PENDGUI MATLAB code for pendgui.fig
%      PENDGUI, by itself, creates a new PENDGUI or raises the existing
%      singleton*.
%
%      H = PENDGUI returns the handle to a new PENDGUI or the handle to
%      the existing singleton*.
%
%      PENDGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PENDGUI.M with the given input arguments.
%
%      PENDGUI('Property','Value',...) creates a new PENDGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before pendgui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to pendgui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help pendgui

% Last Modified by GUIDE v2.5 23-Sep-2014 22:08:14

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @pendgui_OpeningFcn, ...
                   'gui_OutputFcn',  @pendgui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before pendgui is made visible.
function pendgui_OpeningFcn(hObject, ~, handles, varargin)
    clear pendtd
    %mex pendtd.cpp

    handles.output = hObject;
    handles.opt = pendtd('get');
    handles.opt.repetitions = 10;
    handles.opt.view = 0;

    % Slider listeners
    %handles.sliderAlphaListener = addlistener(handles.sliderAlpha,'ContinuousValueChange',@sliderAlpha_Callback);

    update(hObject, handles);

    % UIWAIT makes pendgui wait for user response (see UIRESUME)
    % uiwait(handles.figureRLSandbox);

function update(hObject, handles)
    % Store
    guidata(hObject, handles);
    
    handles.opt.step = max(handles.opt.step, 0.001);
    handles.opt.steps = round(3/handles.opt.step);
    handles.opt.episodes = max(handles.opt.episodes, 1);
    handles.opt.repetitions = max(handles.opt.repetitions, 1);
    handles.opt.observations = max(handles.opt.observations, 1);
    handles.opt.actions = max(handles.opt.actions, 1);

    % Update GUI
    set(handles.sliderAlpha, 'Value', handles.opt.alpha);
    set(handles.editAlpha, 'String', num2str(handles.opt.alpha));
    set(handles.sliderGamma, 'Value', handles.opt.gamma);
    set(handles.editGamma, 'String', num2str(handles.opt.gamma));
    set(handles.sliderEpsilon, 'Value', handles.opt.epsilon);
    set(handles.editEpsilon, 'String', num2str(handles.opt.epsilon));
    set(handles.sliderLambda, 'Value', handles.opt.lambda);
    set(handles.editLambda, 'String', num2str(handles.opt.lambda));

    set(handles.sliderStep, 'Value', handles.opt.step);
    set(handles.editStep, 'String', num2str(handles.opt.step));
    set(handles.sliderEpisodes, 'Value', handles.opt.episodes);
    set(handles.editEpisodes, 'String', num2str(handles.opt.episodes));
    set(handles.sliderRepetitions, 'Value', handles.opt.repetitions);
    set(handles.editRepetitions, 'String', num2str(handles.opt.repetitions));

    set(handles.sliderObservations, 'Value', handles.opt.observations);
    set(handles.editObservations, 'String', num2str(handles.opt.observations));
    set(handles.sliderActions, 'Value', handles.opt.actions);
    set(handles.editActions, 'String', num2str(handles.opt.actions));

    set(handles.sliderGoalWeight, 'Value', handles.opt.goal_weight);
    set(handles.editGoalWeight, 'String', num2str(handles.opt.goal_weight));
    set(handles.sliderQuadraticWeight, 'Value', handles.opt.quadratic_weight);
    set(handles.editQuadraticWeight, 'String', num2str(handles.opt.quadratic_weight));
    set(handles.sliderActionWeight, 'Value', handles.opt.action_weight);
    set(handles.editActionWeight, 'String', num2str(handles.opt.action_weight));
    set(handles.sliderTimeWeight, 'Value', handles.opt.time_weight);
    set(handles.editTimeWeight, 'String', num2str(handles.opt.time_weight));
    
    set(handles.sliderInitial, 'Value', handles.opt.initial);
    set(handles.editInitial, 'String', num2str(handles.opt.initial));

    set(handles.checkboxOnPolicy, 'Value', handles.opt.on_policy);
    set(handles.checkboxReportTests, 'Value', handles.opt.report_tests);
    
    onoff = {'on', 'off'};
    
    set(handles.Learning, 'Checked', onoff{handles.opt.view+1});
    set(handles.Controller, 'Checked', onoff{2-handles.opt.view});
    
    % Run
    pendtd('set', handles.opt);
    [curve, q, path] = pendtd('run');
    for i = 1:handles.opt.repetitions-1
      curve = curve + pendtd('run');
    end
    curve = curve / handles.opt.repetitions;

    if handles.opt.view == 0
        % Display
        plot(handles.axesCurve, squeeze(curve(1,:)));

        mi = min(curve(1,:));

        if mi < -1000
            mi = -10000;
        elseif mi < -100
            mi = -1000;
        elseif mi < -10
            mi = -100;
        elseif mi < -1
            mi = -10;
        else
            mi = 0;
        end

        axis(handles.axesCurve, [0 handles.opt.episodes mi 100])
        [xx, yy] = meshgrid(1:handles.opt.observations, 1:handles.opt.observations);
        surf(handles.axesQ, xx, yy, squeeze(max(q, [], 3)));
        view(0, 90);
        axis image;
    else
        x = 1:handles.opt.episodes;
        plot(handles.axesCurve, x, squeeze(curve(2,:)), x, squeeze(curve(3,:)));
        legend(handles.axesCurve, 'Position error', 'Torque');
        
        x = handles.opt.step.*(0:(size(path, 2)-1));
        plot(handles.axesQ, x, squeeze(path(1,:)), x, squeeze(path(2,:)), x, squeeze(path(3,:)));
        legend(handles.axesQ, 'Position', 'Velocity', 'Torque');
    end

% --- Outputs from this function are returned to the command line.
function varargout = pendgui_OutputFcn(~, ~, handles) 
    % varargout  cell array for returning output args (see VARARGOUT);
    % hObject    handle to figure
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)

    % Get default command line output from handles structure
    varargout{1} = handles.output;

function sliderAlpha_Callback(hObject, ~, ~)
    handles = guidata(hObject);
    handles.opt.alpha = get(hObject, 'Value');
    update(hObject, handles);

function sliderAlpha_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editAlpha_Callback(hObject, ~, handles)
    handles.opt.alpha = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editAlpha_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderGamma_Callback(hObject, ~, handles)
    handles.opt.gamma = get(hObject, 'Value');
    update(hObject, handles);

function sliderGamma_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editGamma_Callback(hObject, ~, handles)
    handles.opt.gamma = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editGamma_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderEpsilon_Callback(hObject, ~, handles)
    handles.opt.epsilon = get(hObject, 'Value');
    update(hObject, handles);

function sliderEpsilon_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editEpsilon_Callback(hObject, ~, handles)
    handles.opt.epsilon = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editEpsilon_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderLambda_Callback(hObject, ~, handles)
    handles.opt.lambda = get(hObject, 'Value');
    update(hObject, handles);

function sliderLambda_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editLambda_Callback(hObject, ~, handles)
    handles.opt.lambda = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editLambda_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderEpisodes_Callback(hObject, ~, handles)
    handles.opt.episodes = get(hObject, 'Value');
    update(hObject, handles);

function sliderEpisodes_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editEpisodes_Callback(hObject, ~, handles)
    handles.opt.episodes = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editEpisodes_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderRepetitions_Callback(hObject, ~, handles)
    handles.opt.repetitions = get(hObject, 'Value');
    update(hObject, handles);

function sliderRepetitions_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editRepetitions_Callback(hObject, ~, handles)
    handles.opt.repetitions = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editRepetitions_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderObservations_Callback(hObject, ~, handles)
    handles.opt.observations = get(hObject, 'Value');
    update(hObject, handles);

function sliderObservations_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editObservations_Callback(hObject, ~, handles)
    handles.opt.observations = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editObservations_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderActions_Callback(hObject, ~, handles)
    handles.opt.actions = get(hObject, 'Value');
    update(hObject, handles);

function sliderActions_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editActions_Callback(hObject, ~, handles)
    handles.opt.actions = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editActions_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderStep_Callback(hObject, ~, handles)
    handles.opt.step = get(hObject, 'Value');
    update(hObject, handles);

function sliderStep_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editStep_Callback(hObject, ~, handles)
    handles.opt.step = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editStep_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderGoalWeight_Callback(hObject, ~, handles)
    handles.opt.goal_weight = get(hObject, 'Value');
    update(hObject, handles);

function sliderGoalWeight_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editGoalWeight_Callback(hObject, ~, handles)
    handles.opt.goal_weight = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editGoalWeight_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderQuadraticWeight_Callback(hObject, ~, handles)
    handles.opt.quadratic_weight = get(hObject, 'Value');
    update(hObject, handles);

function sliderQuadraticWeight_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editQuadraticWeight_Callback(hObject, ~, handles)
    handles.opt.quadratic_weight = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editQuadraticWeight_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderActionWeight_Callback(hObject, ~, handles)
    handles.opt.action_weight = get(hObject, 'Value');
    update(hObject, handles);

function sliderActionWeight_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editActionWeight_Callback(hObject, ~, handles)
    handles.opt.action_weight = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editActionWeight_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderTimeWeight_Callback(hObject, ~, handles)
    handles.opt.time_weight = get(hObject, 'Value');
    update(hObject, handles);

function sliderTimeWeight_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editTimeWeight_Callback(hObject, ~, handles)
    handles.opt.time_weight = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editTimeWeight_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function sliderInitial_Callback(hObject, ~, handles)
    handles.opt.initial = get(hObject, 'Value');
    update(hObject, handles);

function sliderInitial_CreateFcn(hObject, ~, ~)
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end

function editInitial_Callback(hObject, ~, handles)
    handles.opt.initial = str2double(get(hObject, 'String'));
    update(hObject, handles);

function editInitial_CreateFcn(hObject, ~, ~)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

function checkboxOnPolicy_Callback(hObject, ~, handles)
    handles.opt.on_policy = get(hObject, 'Value');
    update(hObject, handles);

function checkboxReportTests_Callback(hObject, ~, handles)
    handles.opt.report_tests = get(hObject, 'Value');
    update(hObject, handles);


% --------------------------------------------------------------------
function File_Callback(~, ~, ~)
% hObject    handle to File (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Reset_Callback(hObject, ~, handles)
% hObject    handle to Reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

clear pendtd
repetitions = handles.opt.repetitions;
view = handles.opt.view;
handles.opt = pendtd('get');
handles.opt.repetitions = repetitions;
handles.opt.view = view;
update(hObject, handles);


% --------------------------------------------------------------------
function View_Callback(hObject, eventdata, handles)
% hObject    handle to View (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Learning_Callback(hObject, eventdata, handles)
% hObject    handle to Learning (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.opt.view = 0;
update(hObject, handles);


% --------------------------------------------------------------------
function Controller_Callback(hObject, eventdata, handles)
% hObject    handle to Controller (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.opt.view = 1;
update(hObject, handles);

