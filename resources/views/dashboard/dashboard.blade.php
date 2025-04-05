@extends('layouts.admin')
@section('page-title')
    {{__('Dashboard')}}
@endsection
@push('script-page')
    <script>
        $(document).ready(function()
        {
            get_data();
        });

        function get_data()
        {
            var calender_type=$('#calender_type :selected').val();
            $('#calendar').removeClass('local_calender');
            $('#calendar').removeClass('goggle_calender');
            if(calender_type==undefined){
                $('#calendar').addClass('local_calender');
            }
            $('#calendar').addClass(calender_type);
            $.ajax({
                url: $("#event_dashboard").val()+"/event/get_event_data" ,
                method:"POST",
                data: {"_token": "{{ csrf_token() }}",'calender_type':calender_type},
                success: function(data) {
                    (function () {
                        var etitle;
                        var etype;
                        var etypeclass;
                        var calendar = new FullCalendar.Calendar(document.getElementById('calendar'), {
                            headerToolbar: {
                                left: 'prev,next today',
                                center: 'title',
                                right: 'timeGridDay,timeGridWeek,dayGridMonth'
                            },
                            buttonText: {
                                timeGridDay: "{{__('Day')}}",
                                timeGridWeek: "{{__('Week')}}",
                                dayGridMonth: "{{__('Month')}}"
                            },
                            slotLabelFormat: {
                                hour: '2-digit',
                                minute: '2-digit',
                                hour12: false,
                            },
                            themeSystem: 'bootstrap',
                            navLinks: true,
                            droppable: true,
                            selectable: true,
                            selectMirror: true,
                            editable: true,
                            dayMaxEvents: true,
                            handleWindowResize: true,
                            height: 'auto',
                            timeFormat: 'H(:mm)',
                            {{--events: {!! json_encode($arrEvents) !!},--}}
                            events: data,
                            locale: '{{basename(App::getLocale())}}',
                            dayClick: function (e) {
                                var t = moment(e).toISOString();
                                $("#new-event").modal("show"), $(".new-event--title").val(""), $(".new-event--start").val(t), $(".new-event--end").val(t)
                            },
                            eventResize: function (event) {
                                var eventObj = {
                                    start: event.start.format(),
                                    end: event.end.format(),
                                };
                            },
                            viewRender: function (t) {
                                e.fullCalendar("getDate").month(), $(".fullcalendar-title").html(t.title)
                            },
                            eventClick: function (e, t) {
                                var title = e.title;
                                var url = e.url;

                                if (typeof url != 'undefined') {
                                    $("#commonModal .modal-title").html(title);
                                    $("#commonModal .modal-dialog").addClass('modal-md');
                                    $("#commonModal").modal('show');
                                    $.get(url, {}, function (data) {
                                        $('#commonModal .modal-body').html(data);
                                    });
                                    return false;
                                }
                            }
                        });
                        calendar.render();
                    })();
                }
            });
        }
    </script>
@endpush
@section('breadcrumb')
    <li class="breadcrumb-item"><a href="{{route('dashboard')}}">{{__('Dashboard')}}</a></li>
    <li class="breadcrumb-item">{{__('HRM')}}</li>
@endsection
@php
    $setting = \App\Models\Utility::settings();
@endphp
@section('content')
    @if(\Auth::user()->type != 'client' && \Auth::user()->type != 'company')
        <div class="row">
            <div class="col-sm-12">
                <div class="row">
                    <div class="col-xxl-6">
                        <div class="card">
                            <div class="card-header">
                                <h4>{{__('Mark Attandance')}}</h4>
                            </div>
                            <div class="card-body dash-card-body">
                                <p class="text-muted pb-0-5">{{__('My Office Time: '.$officeTime['startTime'].' to '.$officeTime['endTime'])}}</p>
                                <center>
                                    <div class="row">
                                        <div class="col-md-6">
                                            {{Form::open(array('url'=>'attendanceemployee/attendance','method'=>'post'))}}
                                            @if(empty($employeeAttendance) || $employeeAttendance->clock_out != '00:00:00')
                                                <button type="submit" value="0" name="in" id="clock_in" class="btn btn-success ">{{__('CLOCK IN')}}</button>
                                            @else
                                                <button type="submit" value="0" name="in" id="clock_in" class="btn btn-success disabled" disabled>{{__('CLOCK IN')}}</button>
                                            @endif
                                            {{Form::close()}}
                                        </div>
                                        <div class="col-md-6 ">
                                            @if(!empty($employeeAttendance) && $employeeAttendance->clock_out == '00:00:00')
                                                {{Form::model($employeeAttendance,array('route'=>array('attendanceemployee.update',$employeeAttendance->id),'method' => 'PUT')) }}
                                                <button type="submit" value="1" name="out" id="clock_out" class="btn btn-danger">{{__('CLOCK OUT')}}</button>
                                            @else
                                                <button type="submit" value="1" name="out" id="clock_out" class="btn btn-danger disabled" disabled>{{__('CLOCK OUT')}}</button>
                                            @endif
                                            {{Form::close()}}
                                        </div>
                                    </div>
                                </center>

                            </div>
                        </div>
                        <div class="card">
                            <div class="card-header">
                                <div class="row">
                                    <div class="col-lg-6">
                                        <h5>{{ __('Event') }}</h5>
                                    </div>
                                    <div class="col-lg-6">
                                        @if (isset($setting['google_calendar_enable']) && $setting['google_calendar_enable'] == 'on')
                                        <select class="form-control" name="calender_type" id="calender_type" onchange="get_data()">
                                            <option value="goggle_calender">{{__('Google Calender')}}</option>
                                            <option value="local_calender" selected="true">{{__('Local Calender')}}</option>
                                        </select>
                                        @endif
                                        <input type="hidden" id="event_dashboard" value="{{url('/')}}">
                                    </div>
                                </div>
                            </div>
                            <div class="card-body">
                                <div id='calendar' class='calendar e-height'></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-xxl-6">
                        <div class="card list_card">
                            <div class="card-header">
                                <h4>{{__('Announcement List')}}</h4>
                            </div>
                            <div class="card-body dash-card-body">
                                <div class="table-responsive">
                                    <table class="table table-striped mb-0">
                                        <thead>
                                        <tr>
                                            <th>{{__('Title')}}</th>
                                            <th>{{__('Start Date')}}</th>
                                            <th>{{__('End Date')}}</th>
                                            <th>{{__('description')}}</th>
                                        </tr>
                                        </thead>
                                        <tbody>
                                        @forelse($announcements as $announcement)
                                            <tr>
                                                <td>{{ $announcement->title }}</td>
                                                <td>{{ \Auth::user()->dateFormat($announcement->start_date)  }}</td>
                                                <td>{{ \Auth::user()->dateFormat($announcement->end_date) }}</td>
                                                <td>{{ $announcement->description }}</td>
                                            </tr>
                                        @empty
                                            <tr>
                                                <td colspan="4">
                                                    <div class="text-center">
                                                        <h6>{{__('There is no Announcement List')}}</h6>
                                                    </div>
                                                </td>
                                            </tr>
                                        @endforelse
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                        <div class="card list_card">
                            <div class="card-header">
                                <h4>{{__('Meeting List')}}</h4>
                            </div>
                            <div class="card-body dash-card-body">
                                @if(count($meetings) > 0)
                                    <div class="table-responsive">
                                        <table class="table align-items-center">
                                            <thead>
                                            <tr>
                                                <th>{{__('Meeting title')}}</th>
                                                <th>{{__('Meeting Date')}}</th>
                                                <th>{{__('Meeting Time')}}</th>
                                            </tr>
                                            </thead>
                                            <tbody>
                                            @forelse($meetings as $meeting)
                                                <tr>
                                                    <td>{{ $meeting->title }}</td>
                                                    <td>{{ \Auth::user()->dateFormat($meeting->date) }}</td>
                                                    <td>{{ \Auth::user()->timeFormat($meeting->time) }}</td>
                                                </tr>
                                                @endforeach
                                            </tbody>
                                        </table>
                                    </div>
                                @else
                                    <div class="p-2">
                                        {{__('No meeting scheduled yet.')}}
                                    </div>
                                @endif
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    @else
        <div class="row">
            <div class="col-xxl-12">
                <div class="card">
                    <div class="card-header">
                        <h5>{{__("Today's Not Clock In")}}</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-12">
                                <div class="d-flex gap-1 team-lists horizontal-scroll-cards">
                                    @foreach($notClockIns as $notClockIn)
                                    @php
                                        $user = $notClockIn->user;
                                        $logo= asset(Storage::url('uploads/avatar/'));
                                        $avatar = !empty($notClockIn->user->avatar) ? $notClockIn->user->avatar : 'avatar.png';
                                    @endphp
                                        <div>
                                            <img src="{{ $logo . $avatar }}" alt="" class="rounded border-2 border border-primary">
                                            <p class="mt-2 p-0">{{ $notClockIn->name }}</p>
                                        </div>
                                    @endforeach
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-sm-12">
                <div class="row">
                    <div class="col-xl-8 mb-4">
                        <div class="card h-100 mb-0">
                            <div class="card-header">
                                <div class="row">
                                    <div class="col-lg-6">
                                        <h5>{{ __('Event') }}</h5>
                                    </div>
                                    <div class="col-lg-6">

                                        @if(isset($setting['google_calendar_enable']) && $setting['google_calendar_enable'] == 'on')
                                            <select class="form-control" name="calender_type" id="calender_type" onchange="get_data()">
                                                <option value="goggle_calender">{{__('Google Calender')}}</option>
                                                <option value="local_calender" selected="true">{{__('Local Calender')}}</option>
                                            </select>
                                        @endif
                                        <input type="hidden" id="event_dashboard" value="{{url('/')}}">
                                    </div>
                                </div>
                            </div>
                            <div class="card-body">
                                <div id='calendar' class='calendar'></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-xl-4">
                        <div class="col-xxl-12">
                            <div class="card">
                                <div class="card-body p-3">
                                    <h4 class="mb-3">{{__('Staff')}}</h4>
                                    <div class="row gy-3">
                                        <div class="col-xxl-6 col-xl-12 col-md-4 col-sm-6 col-12 staff-info-card">
                                            <div class="staff-info-inner d-flex align-items-center gap-3">
                                                <div class="staff-info-icon">
                                                    <svg width="24" height="14" viewBox="0 0 24 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <path d="M11.0423 6.94758C13.2185 7.59464 15.5784 5.86974 15.5442 3.53761C15.3502 -1.15819 8.65015 -1.15791 8.4563 3.53767C8.45631 5.15785 9.5522 6.52972 11.0423 6.94758Z" fill="white"/>
                                                        <path d="M17.7126 12.6242C17.2475 9.89627 14.8625 7.81485 12.0005 7.81485C8.76806 7.79105 6.06201 10.5753 6.22529 13.7359C6.27656 13.8936 6.42636 14 6.59587 14H17.4052C18.0555 13.9452 17.7338 13.0378 17.7126 12.6242Z" fill="white"/>
                                                        <path d="M18.8793 8.12234C20.3063 8.12234 21.4693 6.95941 21.4693 5.53237C21.3392 2.10642 16.4188 2.10735 16.2893 5.5324C16.2893 6.95941 17.4522 8.12234 18.8793 8.12234Z" fill="white"/>
                                                        <path d="M18.8794 8.39832C18.1461 8.39832 17.4208 8.59147 16.79 8.95809C16.9753 9.15125 17.1448 9.36019 17.3025 9.577C18.0136 10.5366 18.4577 11.7291 18.5443 12.9475H22.6441C22.8609 12.9475 23.0383 12.7701 23.0383 12.5533C23.0383 10.2629 21.1737 8.39832 18.8794 8.39832Z" fill="white"/>
                                                        <path d="M5.12147 8.12234C6.54851 8.12234 7.71144 6.95941 7.71144 5.53237C7.58143 2.10642 2.66099 2.10735 2.53149 5.5324C2.53149 6.95941 3.69442 8.12234 5.12147 8.12234Z" fill="white"/>
                                                        <path d="M7.21051 8.95806C4.52252 7.33743 0.906211 9.40945 0.962315 12.5533C0.962268 12.7701 1.13967 12.9475 1.35648 12.9475H5.45629C5.56285 11.4514 6.1974 10.0316 7.21051 8.95806Z" fill="white"/>
                                                    </svg>                                                        
                                                </div>
                                                <div class="staff-info">
                                                    <p class="mb-1 dashboard-link">{{__('Total Staff')}}</p>
                                                    <h4 class="mb-0">{{ $countUser +   $countClient}}</h4>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="col-xxl-6 col-xl-12 col-md-4 col-sm-6 col-12 staff-info-card">
                                            <div class="staff-info-inner d-flex align-items-center gap-3">
                                                <div class="staff-info-icon">
                                                    <svg width="24" height="20" viewBox="0 0 24 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <path d="M9.36694 8.87775C11.7508 8.87775 13.6833 6.94523 13.6833 4.56134C13.6833 2.17746 11.7508 0.244934 9.36694 0.244934C6.98306 0.244934 5.05054 2.17746 5.05054 4.56134C5.05054 6.94523 6.98306 8.87775 9.36694 8.87775Z" fill="white"/>
                                                        <path d="M9.87953 12.7625H8.85438L8.31885 16.7611L9.36696 18.0712L10.4151 16.7611L9.87953 12.7625Z" fill="white"/>
                                                        <path d="M8.83643 11.4676H9.89766L10.1135 10.1727H8.62061L8.83643 11.4676Z" fill="white"/>
                                                        <path d="M6.99866 16.8635L7.63322 12.1255L7.30772 10.1727H6.12964C3.26898 10.1727 0.949951 12.4917 0.949951 15.3524V19.1076C0.949951 19.4652 1.23984 19.7551 1.59741 19.7551H9.05577L7.1348 17.3539C7.02439 17.2159 6.97518 17.0387 6.99866 16.8635Z" fill="white"/>
                                                        <path d="M13.794 10.3108C13.412 10.2209 13.0139 10.1727 12.6044 10.1727H11.4263L11.1008 12.1255L11.7354 16.8635C11.7589 17.0387 11.7097 17.2159 11.5992 17.3539L9.67822 19.7551H13.0378C12.6304 19.2136 12.3885 18.5409 12.3885 17.8127V12.9783C12.3885 11.8727 12.9458 10.8951 13.794 10.3108Z" fill="white"/>
                                                        <path d="M21.1076 11.0359H20.7407V9.95685C20.7407 9.12382 20.063 8.44611 19.2299 8.44611H17.5034C16.6703 8.44611 15.9926 9.12382 15.9926 9.95685V11.0359H15.6257C14.553 11.0359 13.6833 11.9056 13.6833 12.9783V13.41H23.05V12.9783C23.05 11.9056 22.1803 11.0359 21.1076 11.0359ZM19.4458 11.0359H17.2875V9.95685C17.2875 9.83784 17.3844 9.74103 17.5034 9.74103H19.2299C19.3489 9.74103 19.4458 9.83784 19.4458 9.95685V11.0359Z" fill="white"/>
                                                        <path d="M13.6833 17.8127C13.6833 18.8855 14.553 19.7551 15.6257 19.7551H21.1076C22.1803 19.7551 23.05 18.8855 23.05 17.8127V14.7049H13.6833V17.8127Z" fill="white"/>
                                                    </svg>                                                                                                                
                                                </div>
                                                <div class="staff-info">
                                                    <p class="mb-1">
                                                        <a href="{{ route('employee.index') }}" class="dashboard-link">{{__('Total Employee')}}</a>
                                                    </p>
                                                    <h4 class="mb-0">{{$countUser}}</h4>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="col-xxl-6 col-xl-12 col-md-4 col-sm-6 col-12 staff-info-card">
                                            <div class="staff-info-inner d-flex align-items-center gap-3">
                                                <div class="staff-info-icon">
                                                    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <path fill-rule="evenodd" clip-rule="evenodd" d="M11.1357 3.18426L11.8316 5.01718C11.8525 5.0723 11.9012 5.10765 11.9601 5.11054L13.9183 5.20598C13.9796 5.20896 14.0301 5.24759 14.0491 5.30599C14.0681 5.36439 14.0499 5.42534 14.0021 5.4638L12.4739 6.69203C12.428 6.72894 12.4094 6.78622 12.4248 6.84306L12.9392 8.73494C12.9553 8.79421 12.9342 8.85416 12.8845 8.89025C12.8348 8.92633 12.7713 8.92789 12.7199 8.89426L11.0795 7.82047C11.0302 7.78818 10.97 7.78818 10.9207 7.82047L9.28038 8.89426C9.22902 8.92789 9.16544 8.92633 9.1158 8.89025C9.06612 8.85416 9.04501 8.79417 9.06111 8.73494L9.57545 6.84306C9.59091 6.78622 9.5723 6.72894 9.52637 6.69203L7.99824 5.4638C7.95037 5.42534 7.9322 5.36439 7.95119 5.30599C7.97014 5.24759 8.02068 5.209 8.08202 5.20598L10.0402 5.11054C10.0991 5.1077 10.1478 5.0723 10.1687 5.01718L10.8646 3.18426C10.8864 3.12685 10.9387 3.09077 11.0001 3.09077C11.0615 3.09077 11.1139 3.12685 11.1357 3.18426ZM7.34309 2.60763L8.56792 3.71052C8.70945 3.8379 8.92747 3.82641 9.05485 3.68488C9.18223 3.54334 9.17075 3.32532 9.02921 3.19794L7.80439 2.09506C7.66285 1.96768 7.44483 1.97916 7.31745 2.1207C7.19008 2.26223 7.20152 2.48026 7.34309 2.60763ZM14.1958 2.09506L12.9638 3.20438C12.8223 3.33175 12.8108 3.54978 12.9382 3.69131C13.0656 3.83285 13.2836 3.84433 13.4251 3.71695L14.6571 2.60763C14.7987 2.48026 14.8101 2.26223 14.6828 2.1207C14.5553 1.97916 14.3374 1.96768 14.1958 2.09506ZM11.3454 1.80806C11.3454 1.99876 11.1908 2.15337 11.0001 2.15337C10.8094 2.15337 10.6548 1.99876 10.6548 1.80806V0.540808C10.6548 0.350109 10.8094 0.195496 11.0001 0.195496C11.1908 0.195496 11.3454 0.350109 11.3454 0.540808V1.80806ZM11.0001 10.0022C12.3382 10.0022 13.4229 11.087 13.4229 12.425C13.4229 13.763 12.3382 14.8477 11.0001 14.8477C9.66208 14.8477 8.57741 13.763 8.57741 12.425C8.57741 11.0869 9.66208 10.0022 11.0001 10.0022ZM15.7642 19.5555C15.507 17.1516 13.4723 15.2793 11.0001 15.2793C8.52795 15.2793 6.49324 17.1516 6.23598 19.5555V20.524C6.23598 21.2293 6.81123 21.8046 7.51653 21.8046H14.4837C15.189 21.8046 15.7642 21.2293 15.7642 20.524V19.5555ZM18.2408 11.7616C19.2418 11.7616 20.0533 12.5731 20.0533 13.5741C20.0533 14.5751 19.2418 15.3866 18.2408 15.3866C17.2398 15.3866 16.4283 14.5751 16.4283 13.5741C16.4283 12.5731 17.2397 11.7616 18.2408 11.7616ZM3.75944 11.7616C4.76046 11.7616 5.57194 12.5731 5.57194 13.5741C5.57194 14.5751 4.76046 15.3866 3.75944 15.3866C2.75842 15.3866 1.94694 14.5751 1.94694 13.5741C1.94694 12.5731 2.75842 11.7616 3.75944 11.7616ZM3.7594 15.8708C4.81571 15.8708 5.76527 16.3276 6.42137 17.0545C5.95835 17.7556 5.65132 18.5714 5.55239 19.4557H1.11711C0.848326 19.4557 0.617053 19.3388 0.457562 19.1225C0.298071 18.9062 0.254864 18.6507 0.334372 18.3939C0.786947 16.9324 2.14925 15.8708 3.7594 15.8708ZM18.2408 15.8708C19.851 15.8708 21.2133 16.9324 21.6658 18.3939C21.7453 18.6507 21.7021 18.9061 21.5426 19.1225C21.3831 19.3389 21.1519 19.4557 20.8831 19.4557H16.4478C16.3488 18.5714 16.0418 17.7556 15.5788 17.0545C16.2349 16.3276 17.1845 15.8708 18.2408 15.8708ZM3.48729 3.18426L4.18318 5.01718C4.20412 5.0723 4.25281 5.10765 4.31168 5.11054L6.26991 5.20598C6.33124 5.20896 6.38174 5.24759 6.40074 5.30599C6.41968 5.36439 6.40156 5.42534 6.35369 5.4638L4.82555 6.69203C4.77962 6.72894 4.76102 6.78622 4.77647 6.84306L5.29081 8.73494C5.30692 8.79421 5.28581 8.85416 5.23613 8.89025C5.18644 8.92633 5.12291 8.92789 5.0715 8.89426L3.43113 7.82047C3.38184 7.78818 3.32163 7.78818 3.27233 7.82047L1.63201 8.89426C1.58065 8.92789 1.51707 8.92633 1.46743 8.89025C1.41775 8.85416 1.39664 8.79417 1.41274 8.73494L1.92708 6.84306C1.94254 6.78622 1.92393 6.72894 1.87801 6.69203L0.349868 5.46376C0.301999 5.4253 0.283827 5.36435 0.302819 5.30595C0.321768 5.24755 0.372313 5.20896 0.433649 5.20594L2.39187 5.1105C2.45075 5.10765 2.49944 5.07226 2.52033 5.01714L3.21626 3.18426C3.23806 3.12685 3.29038 3.09077 3.3518 3.09077C3.41322 3.09077 3.46549 3.12685 3.48729 3.18426ZM18.7612 3.18426L19.457 5.01718C19.478 5.0723 19.5267 5.10765 19.5855 5.11054L21.5438 5.20598C21.6051 5.20896 21.6556 5.24759 21.6746 5.30599C21.6935 5.36439 21.6754 5.42534 21.6276 5.4638L20.0994 6.69203C20.0535 6.72894 20.0349 6.78622 20.0503 6.84306L20.5647 8.73494C20.5808 8.79421 20.5597 8.85416 20.51 8.89025C20.4604 8.92633 20.3968 8.92789 20.3454 8.89426L18.705 7.82047C18.6557 7.78818 18.5955 7.78818 18.5462 7.82047L16.9059 8.89426C16.8546 8.92789 16.791 8.92633 16.7413 8.89025C16.6916 8.85416 16.6705 8.79417 16.6866 8.73494L17.2009 6.84306C17.2164 6.78622 17.1978 6.72894 17.1519 6.69203L15.6237 5.4638C15.5759 5.42534 15.5577 5.36439 15.5767 5.30599C15.5956 5.24759 15.6462 5.209 15.7075 5.20598L17.6657 5.11054C17.7246 5.1077 17.7733 5.0723 17.7942 5.01718L18.4901 3.18426C18.5119 3.12685 18.5642 3.09077 18.6256 3.09077C18.687 3.09077 18.7394 3.12685 18.7612 3.18426Z" fill="white"/>
                                                    </svg>                                                                                                               
                                                </div>
                                                <div class="staff-info">
                                                    <p class="mb-1">
                                                        <a href="{{ route('clients.index') }}" class="dashboard-link">{{__('Total Client')}}</a>
                                                    </p>
                                                    <h4 class="mb-0">{{$countClient}}</h4>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-xxl-12">
                            <div class="card">
                                <div class="card-body p-3">
                                    <h4 class="mb-3">{{__('Job')}}</h4>
                                    <div class="row gy-3">
                                        <div class="col-xxl-6 col-xl-12 col-md-4 col-sm-6 col-12 staff-info-card">
                                            <div class="staff-info-inner d-flex align-items-center gap-3">
                                                <div class="staff-info-icon">
                                                    <svg width="24" height="20" viewBox="0 0 24 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <path d="M12.6475 13.8848C13.005 13.8848 13.2949 13.5949 13.2949 13.2373V10.6475C13.2949 10.2899 13.0051 10 12.6475 10H11.3525C10.995 10 10.7051 10.2898 10.7051 10.6475V13.2373C10.7051 13.5948 10.995 13.8848 11.3525 13.8848H12.6475Z" fill="white"/>
                                                        <path d="M21.1076 2.87793H16.5322C16.5322 2.74753 16.5322 2.10007 16.5322 2.23047C16.5322 1.1594 15.6609 0.288086 14.5898 0.288086C14.4493 0.288086 9.21781 0.288086 9.41011 0.288086C8.33903 0.288086 7.46772 1.15935 7.46772 2.23047C7.46772 2.36087 7.46772 3.00833 7.46772 2.87793H2.89233C1.82126 2.87793 0.949951 3.7492 0.949951 4.82031C0.949951 6.08834 1.10534 7.19883 1.41388 8.1452C1.72242 9.09157 2.18414 9.87388 2.79685 10.4856C3.82791 11.5153 5.14122 11.9424 6.61588 11.9424H9.41011C9.41011 11.812 9.41011 10.5171 9.41011 10.6475C9.41011 9.57639 10.2814 8.70508 11.3525 8.70508C11.4829 8.70508 12.7778 8.70508 12.6474 8.70508C13.7185 8.70508 14.5898 9.57635 14.5898 10.6475C14.5898 10.7779 14.5898 12.0728 14.5898 11.9424H16.0284C17.3173 11.8888 19.4256 12.2807 21.2024 10.5121C21.8154 9.90197 22.2773 9.11829 22.5859 8.16747C22.8946 7.21665 23.05 6.09879 23.05 4.82031C23.05 3.74924 22.1787 2.87793 21.1076 2.87793ZM8.76265 2.23047C8.76265 1.87324 9.05284 1.58301 9.41011 1.58301H14.5898C14.947 1.58301 15.2373 1.8732 15.2373 2.23047C15.2373 2.36087 15.2373 3.00833 15.2373 2.87793H8.76265C8.76265 2.74753 8.76265 2.10007 8.76265 2.23047Z" fill="white"/>
                                                        <path d="M16.0461 13.2373H14.5898C14.5898 14.3084 13.7185 15.1797 12.6474 15.1797C12.517 15.1797 11.2221 15.1797 11.3525 15.1797C10.2814 15.1797 9.41011 14.3084 9.41011 13.2373C9.26218 13.2373 6.46166 13.2373 6.62154 13.2373C4.77481 13.2373 3.15274 12.6707 1.88191 11.4018C1.52291 11.0434 1.21636 10.6399 0.949951 10.2083V19.0645C0.949951 19.4223 1.23954 19.7119 1.59741 19.7119H22.4025C22.7604 19.7119 23.05 19.4223 23.05 19.0645V10.2349C22.78 10.6731 22.4722 11.0757 22.1161 11.4303C20.0731 13.4626 17.8973 13.1499 16.0461 13.2373Z" fill="white"/>
                                                    </svg>                                                                                                               
                                                </div>
                                                <div class="staff-info">
                                                    <p class="mb-1">
                                                        <a href="{{ route('job.index') }}" class="dashboard-link">{{__('Total Jobs')}}</a>
                                                    </p>
                                                    <h4 class="mb-0">{{$activeJob + $inActiveJOb}}</h4>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="col-xxl-6 col-xl-12 col-md-4 col-sm-6 col-12 staff-info-card">
                                            <div class="staff-info-inner d-flex align-items-center gap-3">
                                                <div class="staff-info-icon">
                                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <g clip-path="url(#clip0_27_571)">
                                                        <path d="M17.8271 15.2805C15.6816 15.2805 13.9424 17.0197 13.9424 19.1653C13.9424 21.3108 15.6816 23.05 17.8271 23.05C19.9727 23.05 21.7119 21.3108 21.7119 19.1653C21.7119 17.0197 19.9727 15.2805 17.8271 15.2805ZM17.1797 21.3757L15.427 19.623L16.3425 18.7075L17.1797 19.5446L19.3118 17.4126L20.2273 18.3281L17.1797 21.3757Z" fill="white"/>
                                                        <path d="M12.0718 9.34121C11.2912 9.81315 10.3869 10.1008 9.41016 10.1008C8.42615 10.1008 7.51498 9.80995 6.73078 9.3316C4.1261 10.3906 2.28809 12.9431 2.28809 15.9279V20.4602H12.8314C12.7232 20.0439 12.6475 19.6149 12.6475 19.1653C12.6475 16.8364 14.2027 14.8849 16.3213 14.2351C15.7753 11.9992 14.1687 10.1924 12.0718 9.34121Z" fill="white"/>
                                                        <path d="M13.2949 4.92111C13.2949 2.77555 11.5557 0.950012 9.41016 0.950012C7.2646 0.950012 5.52539 2.77555 5.52539 4.92111C5.52539 7.06666 7.2646 8.80587 9.41016 8.80587C11.5557 8.80587 13.2949 7.06666 13.2949 4.92111Z" fill="white"/>
                                                        </g>
                                                        <defs>
                                                        <clipPath id="clip0_27_571">
                                                        <rect width="22.1" height="22.1" fill="white" transform="translate(0.949951 0.950012)"/>
                                                        </clipPath>
                                                        </defs>
                                                    </svg>                                                                                                                                                                       
                                                </div>
                                                <div class="staff-info">
                                                    <p class="mb-1 dashboard-link">{{__('Active Jobs')}}</p>
                                                    <h4 class="mb-0">{{$activeJob}}</h4>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="col-xxl-6 col-xl-12 col-md-4 col-sm-6 col-12 staff-info-card">
                                            <div class="staff-info-inner d-flex align-items-center gap-3">
                                                <div class="staff-info-icon">
                                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <path fill-rule="evenodd" clip-rule="evenodd" d="M15.5365 3.76293L20.2388 8.46528C20.2963 8.52279 20.3251 8.59819 20.3251 8.6736L20.3251 15.3263C20.3251 15.4086 20.2914 15.483 20.2369 15.5365L15.5346 20.2389C15.477 20.2964 15.4017 20.3252 15.3263 20.3252L8.67352 20.3252C8.59122 20.3252 8.51679 20.2915 8.4633 20.2371L3.76095 15.5347C3.70341 15.4772 3.67467 15.4018 3.67467 15.3264L3.67463 8.67369C3.67463 8.59138 3.7084 8.51695 3.76285 8.46347L8.46516 3.76108C8.52266 3.70353 8.59807 3.6748 8.67347 3.6748L15.3262 3.67475C15.4086 3.67475 15.483 3.70852 15.5365 3.76293ZM3.41451 5.27718V3.56991C3.41451 3.54825 3.43215 3.53062 3.45381 3.53062H5.3042C5.46694 3.53062 5.59888 3.39868 5.59888 3.23594C5.59888 3.07319 5.46694 2.94125 5.3042 2.94125H3.11988C2.95713 2.94125 2.8252 3.07319 2.8252 3.23594V5.27718C2.8252 5.43992 2.95713 5.57186 3.11988 5.57186C3.28262 5.57186 3.41451 5.43992 3.41451 5.27718ZM2.8252 18.7228V20.764C2.8252 20.9268 2.95713 21.0587 3.11988 21.0587H5.3042C5.46694 21.0587 5.59888 20.9268 5.59888 20.764C5.59888 20.6013 5.46694 20.4693 5.3042 20.4693H3.45381C3.43215 20.4693 3.41451 20.4517 3.41451 20.43V18.7228C3.41451 18.5601 3.28258 18.4281 3.11983 18.4281C2.95709 18.4281 2.8252 18.5601 2.8252 18.7228ZM20.5852 18.7228V20.4301C20.5852 20.4517 20.5676 20.4694 20.5459 20.4694H18.6956C18.5328 20.4694 18.4009 20.6013 18.4009 20.7641C18.4009 20.9268 18.5328 21.0587 18.6956 21.0587H20.8799C21.0426 21.0587 21.1746 20.9268 21.1746 20.7641V18.7228C21.1746 18.5601 21.0426 18.4281 20.8799 18.4281C20.7171 18.4281 20.5852 18.5601 20.5852 18.7228ZM21.1746 5.27718V3.23598C21.1746 3.07324 21.0426 2.9413 20.8799 2.9413H18.6956C18.5328 2.9413 18.4009 3.07324 18.4009 3.23598C18.4009 3.39872 18.5328 3.53066 18.6956 3.53066H20.5459C20.5676 3.53066 20.5852 3.5483 20.5852 3.56995V5.27718C20.5852 5.43992 20.7172 5.57186 20.8799 5.57186C21.0427 5.57186 21.1746 5.43992 21.1746 5.27718ZM16.1783 8.66564L15.3342 7.82155C15.175 7.6623 14.9126 7.6623 14.7533 7.82155L12.6772 9.89772C12.3038 10.2711 11.696 10.2711 11.3226 9.89772L9.24644 7.82155C9.08719 7.6623 8.82477 7.6623 8.66552 7.82155L7.82143 8.66564C7.66218 8.82489 7.66218 9.08731 7.82143 9.24656L9.89759 11.3228C10.271 11.6961 10.271 12.3039 9.89759 12.6773L7.82139 14.7535C7.66213 14.9128 7.66213 15.1752 7.82139 15.3344L8.66547 16.1785C8.82473 16.3378 9.08714 16.3378 9.24639 16.1785L11.3226 14.1023C11.696 13.729 12.3038 13.729 12.6772 14.1023L14.7534 16.1785C14.9126 16.3378 15.175 16.3378 15.3343 16.1785L16.1784 15.3344C16.3376 15.1752 16.3376 14.9128 16.1784 14.7535L14.1022 12.6774C13.7288 12.304 13.7288 11.6962 14.1022 11.3228L16.1784 9.24665C16.3375 9.08731 16.3375 8.82489 16.1783 8.66564Z" fill="white"/>
                                                    </svg>                                                                                                                                                                      
                                                </div>
                                                <div class="staff-info">
                                                    <p class="mb-1 dashboard-link">{{__('Inactive Jobs')}}</p>
                                                    <h4 class="mb-0">{{$inActiveJOb}}</h4>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-xxl-12">
                            <div class="card">
                                <div class="card-body p-3">
                                    <h4 class="mb-3">{{__('Training')}}</h4>
                                    <div class="row gy-3">
                                        <div class="col-xxl-6 col-xl-12 col-md-4 col-sm-6 col-12 staff-info-card">
                                            <div class="staff-info-inner d-flex align-items-center gap-3">
                                                <div class="staff-info-icon">
                                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <g clip-path="url(#clip0_27_598)">
                                                        <path d="M21.0155 2.33125H11.3065C10.1847 2.33125 9.27202 3.24394 9.27202 4.3657V10.594L7.46924 9.69747C7.40837 9.66712 7.34649 9.63846 7.28411 9.61131C7.81337 9.05878 8.13963 8.31015 8.13963 7.48649C8.13963 5.79112 6.76041 4.41173 5.06504 4.41173C3.3695 4.41173 1.99027 5.79112 1.99027 7.48649C1.99027 8.40778 2.39814 9.23531 3.04223 9.79931C1.85606 10.4667 0.949951 11.7685 0.949951 13.3813V15.4617C0.949951 16.3574 1.53149 17.1197 2.33693 17.3906V21.0098C2.33693 21.3673 2.62677 21.6573 2.98439 21.6573H7.14534C7.50296 21.6573 7.7928 21.3673 7.7928 21.0098V13.7352L9.86569 14.7714C10.659 15.168 11.7351 14.9316 12.2217 14.0287H14.5431L12.0849 20.7884C11.9627 21.1245 12.1362 21.4961 12.4722 21.6182C12.8563 21.7579 13.1979 21.517 13.302 21.231L15.921 14.0287H16.3749L19.0207 21.2331C19.1212 21.5064 19.4603 21.7613 19.8518 21.6175C20.1873 21.4942 20.3596 21.1223 20.2362 20.7866L17.7543 14.0287H21.0155C22.1155 14.0287 23.05 13.1413 23.05 11.9943V4.3657C23.05 3.21848 22.1155 2.33125 21.0155 2.33125ZM3.28519 7.48649C3.28519 6.50502 4.08356 5.70665 5.06487 5.70665C6.04635 5.70665 6.84471 6.50502 6.84471 7.48649C6.84471 8.4678 6.04635 9.26617 5.06487 9.26617C4.08356 9.26617 3.28519 8.4678 3.28519 7.48649ZM11.1151 13.3364C11.1134 13.3411 11.1117 13.346 11.1102 13.3509C11.066 13.4742 10.9662 13.578 10.844 13.6283C10.7146 13.6809 10.5691 13.6753 10.4449 13.6133C9.21368 12.9973 7.43468 12.1086 7.43468 12.1086C7.00506 11.8938 6.49788 12.2067 6.49788 12.6878V20.3624H3.63186V16.8487C3.63186 16.4913 3.34185 16.2012 2.98439 16.2012C2.57653 16.2012 2.24487 15.8696 2.24487 15.4617V13.3813C2.24487 11.7992 3.53305 10.5613 5.06487 10.5613H5.63696C6.07096 10.5613 6.50446 10.6631 6.89175 10.8565C6.89175 10.8565 10.1033 12.4536 10.9212 12.8609C11.0964 12.9479 11.1797 13.1523 11.1151 13.3364ZM21.755 11.9943C21.755 12.192 21.6783 12.3777 21.5387 12.5173C21.4013 12.6549 21.2106 12.7338 21.0155 12.7338H12.3625C12.3613 12.7303 12.3604 12.7267 12.3593 12.7233L13.9661 9.50981C14.1259 9.18996 13.9965 8.80114 13.6766 8.64113C13.3569 8.48129 12.9679 8.61078 12.8079 8.93063L11.4375 11.6714C11.182 11.5441 10.8851 11.3962 10.5669 11.2381V4.3657C10.5669 3.958 10.8986 3.62617 11.3065 3.62617H21.0155C21.2106 3.62617 21.4013 3.70508 21.5389 3.84301C21.6783 3.98228 21.755 4.16792 21.755 4.3657V11.9943Z" fill="white"/>
                                                        <path d="M19.6284 5.10521H12.6934C12.3357 5.10521 12.0459 5.39505 12.0459 5.75267C12.0459 6.11029 12.3357 6.40013 12.6934 6.40013H19.6284C19.9859 6.40013 20.2759 6.11029 20.2759 5.75267C20.2759 5.39505 19.9861 5.10521 19.6284 5.10521Z" fill="white"/>
                                                        <path d="M19.6285 7.53252H16.1609C15.8034 7.53252 15.5134 7.82236 15.5134 8.17998C15.5134 8.5376 15.8034 8.82744 16.1609 8.82744H19.6285C19.986 8.82744 20.276 8.5376 20.276 8.17998C20.276 7.82236 19.9861 7.53252 19.6285 7.53252Z" fill="white"/>
                                                        <path d="M19.6285 9.95982H16.1609C15.8034 9.95982 15.5134 10.2497 15.5134 10.6073C15.5134 10.9647 15.8034 11.2547 16.1609 11.2547H19.6285C19.986 11.2547 20.276 10.9647 20.276 10.6073C20.276 10.2497 19.9861 9.95982 19.6285 9.95982Z" fill="white"/>
                                                        </g>
                                                        <defs>
                                                        <clipPath id="clip0_27_598">
                                                        <rect width="22.1" height="22.1" fill="white" transform="translate(0.949951 0.949997)"/>
                                                        </clipPath>
                                                        </defs>
                                                    </svg>                                                                                                                                                                      
                                                </div>
                                                <div class="staff-info">
                                                    <p class="mb-1">
                                                        <a href="{{ route('training.index') }}" class="dashboard-link">{{__('Total Training')}}</a>
                                                    </p>
                                                    <h4 class="mb-0">{{ $onGoingTraining + $doneTraining}}</h4>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="col-xxl-6 col-xl-12 col-md-4 col-sm-6 col-12 staff-info-card">
                                            <div class="staff-info-inner d-flex align-items-center gap-3">
                                                <div class="staff-info-icon">
                                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <g clip-path="url(#clip0_27_598)">
                                                        <path d="M21.0155 2.33125H11.3065C10.1847 2.33125 9.27202 3.24394 9.27202 4.3657V10.594L7.46924 9.69747C7.40837 9.66712 7.34649 9.63846 7.28411 9.61131C7.81337 9.05878 8.13963 8.31015 8.13963 7.48649C8.13963 5.79112 6.76041 4.41173 5.06504 4.41173C3.3695 4.41173 1.99027 5.79112 1.99027 7.48649C1.99027 8.40778 2.39814 9.23531 3.04223 9.79931C1.85606 10.4667 0.949951 11.7685 0.949951 13.3813V15.4617C0.949951 16.3574 1.53149 17.1197 2.33693 17.3906V21.0098C2.33693 21.3673 2.62677 21.6573 2.98439 21.6573H7.14534C7.50296 21.6573 7.7928 21.3673 7.7928 21.0098V13.7352L9.86569 14.7714C10.659 15.168 11.7351 14.9316 12.2217 14.0287H14.5431L12.0849 20.7884C11.9627 21.1245 12.1362 21.4961 12.4722 21.6182C12.8563 21.7579 13.1979 21.517 13.302 21.231L15.921 14.0287H16.3749L19.0207 21.2331C19.1212 21.5064 19.4603 21.7613 19.8518 21.6175C20.1873 21.4942 20.3596 21.1223 20.2362 20.7866L17.7543 14.0287H21.0155C22.1155 14.0287 23.05 13.1413 23.05 11.9943V4.3657C23.05 3.21848 22.1155 2.33125 21.0155 2.33125ZM3.28519 7.48649C3.28519 6.50502 4.08356 5.70665 5.06487 5.70665C6.04635 5.70665 6.84471 6.50502 6.84471 7.48649C6.84471 8.4678 6.04635 9.26617 5.06487 9.26617C4.08356 9.26617 3.28519 8.4678 3.28519 7.48649ZM11.1151 13.3364C11.1134 13.3411 11.1117 13.346 11.1102 13.3509C11.066 13.4742 10.9662 13.578 10.844 13.6283C10.7146 13.6809 10.5691 13.6753 10.4449 13.6133C9.21368 12.9973 7.43468 12.1086 7.43468 12.1086C7.00506 11.8938 6.49788 12.2067 6.49788 12.6878V20.3624H3.63186V16.8487C3.63186 16.4913 3.34185 16.2012 2.98439 16.2012C2.57653 16.2012 2.24487 15.8696 2.24487 15.4617V13.3813C2.24487 11.7992 3.53305 10.5613 5.06487 10.5613H5.63696C6.07096 10.5613 6.50446 10.6631 6.89175 10.8565C6.89175 10.8565 10.1033 12.4536 10.9212 12.8609C11.0964 12.9479 11.1797 13.1523 11.1151 13.3364ZM21.755 11.9943C21.755 12.192 21.6783 12.3777 21.5387 12.5173C21.4013 12.6549 21.2106 12.7338 21.0155 12.7338H12.3625C12.3613 12.7303 12.3604 12.7267 12.3593 12.7233L13.9661 9.50981C14.1259 9.18996 13.9965 8.80114 13.6766 8.64113C13.3569 8.48129 12.9679 8.61078 12.8079 8.93063L11.4375 11.6714C11.182 11.5441 10.8851 11.3962 10.5669 11.2381V4.3657C10.5669 3.958 10.8986 3.62617 11.3065 3.62617H21.0155C21.2106 3.62617 21.4013 3.70508 21.5389 3.84301C21.6783 3.98228 21.755 4.16792 21.755 4.3657V11.9943Z" fill="white"/>
                                                        <path d="M19.6284 5.10521H12.6934C12.3357 5.10521 12.0459 5.39505 12.0459 5.75267C12.0459 6.11029 12.3357 6.40013 12.6934 6.40013H19.6284C19.9859 6.40013 20.2759 6.11029 20.2759 5.75267C20.2759 5.39505 19.9861 5.10521 19.6284 5.10521Z" fill="white"/>
                                                        <path d="M19.6285 7.53252H16.1609C15.8034 7.53252 15.5134 7.82236 15.5134 8.17998C15.5134 8.5376 15.8034 8.82744 16.1609 8.82744H19.6285C19.986 8.82744 20.276 8.5376 20.276 8.17998C20.276 7.82236 19.9861 7.53252 19.6285 7.53252Z" fill="white"/>
                                                        <path d="M19.6285 9.95982H16.1609C15.8034 9.95982 15.5134 10.2497 15.5134 10.6073C15.5134 10.9647 15.8034 11.2547 16.1609 11.2547H19.6285C19.986 11.2547 20.276 10.9647 20.276 10.6073C20.276 10.2497 19.9861 9.95982 19.6285 9.95982Z" fill="white"/>
                                                        </g>
                                                        <defs>
                                                        <clipPath id="clip0_27_598">
                                                        <rect width="22.1" height="22.1" fill="white" transform="translate(0.949951 0.949997)"/>
                                                        </clipPath>
                                                        </defs>
                                                    </svg>                                                                                                                                                                                                                              
                                                </div>
                                                <div class="staff-info">
                                                    <p class="mb-1">
                                                        <a href="{{ route('trainer.index') }}" class="dashboard-link">{{__('Trainer')}}</a>
                                                    </p>
                                                    <h4 class="mb-0">{{$countTrainer}}</h4>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="col-xxl-6 col-xl-12 col-md-4 col-sm-6 col-12 staff-info-card">
                                            <div class="staff-info-inner d-flex align-items-center gap-3">
                                                <div class="staff-info-icon">
                                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <g clip-path="url(#clip0_27_598)">
                                                        <path d="M21.0155 2.33125H11.3065C10.1847 2.33125 9.27202 3.24394 9.27202 4.3657V10.594L7.46924 9.69747C7.40837 9.66712 7.34649 9.63846 7.28411 9.61131C7.81337 9.05878 8.13963 8.31015 8.13963 7.48649C8.13963 5.79112 6.76041 4.41173 5.06504 4.41173C3.3695 4.41173 1.99027 5.79112 1.99027 7.48649C1.99027 8.40778 2.39814 9.23531 3.04223 9.79931C1.85606 10.4667 0.949951 11.7685 0.949951 13.3813V15.4617C0.949951 16.3574 1.53149 17.1197 2.33693 17.3906V21.0098C2.33693 21.3673 2.62677 21.6573 2.98439 21.6573H7.14534C7.50296 21.6573 7.7928 21.3673 7.7928 21.0098V13.7352L9.86569 14.7714C10.659 15.168 11.7351 14.9316 12.2217 14.0287H14.5431L12.0849 20.7884C11.9627 21.1245 12.1362 21.4961 12.4722 21.6182C12.8563 21.7579 13.1979 21.517 13.302 21.231L15.921 14.0287H16.3749L19.0207 21.2331C19.1212 21.5064 19.4603 21.7613 19.8518 21.6175C20.1873 21.4942 20.3596 21.1223 20.2362 20.7866L17.7543 14.0287H21.0155C22.1155 14.0287 23.05 13.1413 23.05 11.9943V4.3657C23.05 3.21848 22.1155 2.33125 21.0155 2.33125ZM3.28519 7.48649C3.28519 6.50502 4.08356 5.70665 5.06487 5.70665C6.04635 5.70665 6.84471 6.50502 6.84471 7.48649C6.84471 8.4678 6.04635 9.26617 5.06487 9.26617C4.08356 9.26617 3.28519 8.4678 3.28519 7.48649ZM11.1151 13.3364C11.1134 13.3411 11.1117 13.346 11.1102 13.3509C11.066 13.4742 10.9662 13.578 10.844 13.6283C10.7146 13.6809 10.5691 13.6753 10.4449 13.6133C9.21368 12.9973 7.43468 12.1086 7.43468 12.1086C7.00506 11.8938 6.49788 12.2067 6.49788 12.6878V20.3624H3.63186V16.8487C3.63186 16.4913 3.34185 16.2012 2.98439 16.2012C2.57653 16.2012 2.24487 15.8696 2.24487 15.4617V13.3813C2.24487 11.7992 3.53305 10.5613 5.06487 10.5613H5.63696C6.07096 10.5613 6.50446 10.6631 6.89175 10.8565C6.89175 10.8565 10.1033 12.4536 10.9212 12.8609C11.0964 12.9479 11.1797 13.1523 11.1151 13.3364ZM21.755 11.9943C21.755 12.192 21.6783 12.3777 21.5387 12.5173C21.4013 12.6549 21.2106 12.7338 21.0155 12.7338H12.3625C12.3613 12.7303 12.3604 12.7267 12.3593 12.7233L13.9661 9.50981C14.1259 9.18996 13.9965 8.80114 13.6766 8.64113C13.3569 8.48129 12.9679 8.61078 12.8079 8.93063L11.4375 11.6714C11.182 11.5441 10.8851 11.3962 10.5669 11.2381V4.3657C10.5669 3.958 10.8986 3.62617 11.3065 3.62617H21.0155C21.2106 3.62617 21.4013 3.70508 21.5389 3.84301C21.6783 3.98228 21.755 4.16792 21.755 4.3657V11.9943Z" fill="white"/>
                                                        <path d="M19.6284 5.10521H12.6934C12.3357 5.10521 12.0459 5.39505 12.0459 5.75267C12.0459 6.11029 12.3357 6.40013 12.6934 6.40013H19.6284C19.9859 6.40013 20.2759 6.11029 20.2759 5.75267C20.2759 5.39505 19.9861 5.10521 19.6284 5.10521Z" fill="white"/>
                                                        <path d="M19.6285 7.53252H16.1609C15.8034 7.53252 15.5134 7.82236 15.5134 8.17998C15.5134 8.5376 15.8034 8.82744 16.1609 8.82744H19.6285C19.986 8.82744 20.276 8.5376 20.276 8.17998C20.276 7.82236 19.9861 7.53252 19.6285 7.53252Z" fill="white"/>
                                                        <path d="M19.6285 9.95982H16.1609C15.8034 9.95982 15.5134 10.2497 15.5134 10.6073C15.5134 10.9647 15.8034 11.2547 16.1609 11.2547H19.6285C19.986 11.2547 20.276 10.9647 20.276 10.6073C20.276 10.2497 19.9861 9.95982 19.6285 9.95982Z" fill="white"/>
                                                        </g>
                                                        <defs>
                                                        <clipPath id="clip0_27_598">
                                                        <rect width="22.1" height="22.1" fill="white" transform="translate(0.949951 0.949997)"/>
                                                        </clipPath>
                                                        </defs>
                                                    </svg>                                                                                                                                                                                                                             
                                                </div>
                                                <div class="staff-info">
                                                    <p class="mb-1 dashboard-link">{{__('Active Training')}}</p>
                                                    <h4 class="mb-0">{{$onGoingTraining}}</h4>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="col-xxl-6 col-xl-12 col-md-4 col-sm-6 col-12 staff-info-card">
                                            <div class="staff-info-inner d-flex align-items-center gap-3">
                                                <div class="staff-info-icon">
                                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <g clip-path="url(#clip0_27_598)">
                                                        <path d="M21.0155 2.33125H11.3065C10.1847 2.33125 9.27202 3.24394 9.27202 4.3657V10.594L7.46924 9.69747C7.40837 9.66712 7.34649 9.63846 7.28411 9.61131C7.81337 9.05878 8.13963 8.31015 8.13963 7.48649C8.13963 5.79112 6.76041 4.41173 5.06504 4.41173C3.3695 4.41173 1.99027 5.79112 1.99027 7.48649C1.99027 8.40778 2.39814 9.23531 3.04223 9.79931C1.85606 10.4667 0.949951 11.7685 0.949951 13.3813V15.4617C0.949951 16.3574 1.53149 17.1197 2.33693 17.3906V21.0098C2.33693 21.3673 2.62677 21.6573 2.98439 21.6573H7.14534C7.50296 21.6573 7.7928 21.3673 7.7928 21.0098V13.7352L9.86569 14.7714C10.659 15.168 11.7351 14.9316 12.2217 14.0287H14.5431L12.0849 20.7884C11.9627 21.1245 12.1362 21.4961 12.4722 21.6182C12.8563 21.7579 13.1979 21.517 13.302 21.231L15.921 14.0287H16.3749L19.0207 21.2331C19.1212 21.5064 19.4603 21.7613 19.8518 21.6175C20.1873 21.4942 20.3596 21.1223 20.2362 20.7866L17.7543 14.0287H21.0155C22.1155 14.0287 23.05 13.1413 23.05 11.9943V4.3657C23.05 3.21848 22.1155 2.33125 21.0155 2.33125ZM3.28519 7.48649C3.28519 6.50502 4.08356 5.70665 5.06487 5.70665C6.04635 5.70665 6.84471 6.50502 6.84471 7.48649C6.84471 8.4678 6.04635 9.26617 5.06487 9.26617C4.08356 9.26617 3.28519 8.4678 3.28519 7.48649ZM11.1151 13.3364C11.1134 13.3411 11.1117 13.346 11.1102 13.3509C11.066 13.4742 10.9662 13.578 10.844 13.6283C10.7146 13.6809 10.5691 13.6753 10.4449 13.6133C9.21368 12.9973 7.43468 12.1086 7.43468 12.1086C7.00506 11.8938 6.49788 12.2067 6.49788 12.6878V20.3624H3.63186V16.8487C3.63186 16.4913 3.34185 16.2012 2.98439 16.2012C2.57653 16.2012 2.24487 15.8696 2.24487 15.4617V13.3813C2.24487 11.7992 3.53305 10.5613 5.06487 10.5613H5.63696C6.07096 10.5613 6.50446 10.6631 6.89175 10.8565C6.89175 10.8565 10.1033 12.4536 10.9212 12.8609C11.0964 12.9479 11.1797 13.1523 11.1151 13.3364ZM21.755 11.9943C21.755 12.192 21.6783 12.3777 21.5387 12.5173C21.4013 12.6549 21.2106 12.7338 21.0155 12.7338H12.3625C12.3613 12.7303 12.3604 12.7267 12.3593 12.7233L13.9661 9.50981C14.1259 9.18996 13.9965 8.80114 13.6766 8.64113C13.3569 8.48129 12.9679 8.61078 12.8079 8.93063L11.4375 11.6714C11.182 11.5441 10.8851 11.3962 10.5669 11.2381V4.3657C10.5669 3.958 10.8986 3.62617 11.3065 3.62617H21.0155C21.2106 3.62617 21.4013 3.70508 21.5389 3.84301C21.6783 3.98228 21.755 4.16792 21.755 4.3657V11.9943Z" fill="white"/>
                                                        <path d="M19.6284 5.10521H12.6934C12.3357 5.10521 12.0459 5.39505 12.0459 5.75267C12.0459 6.11029 12.3357 6.40013 12.6934 6.40013H19.6284C19.9859 6.40013 20.2759 6.11029 20.2759 5.75267C20.2759 5.39505 19.9861 5.10521 19.6284 5.10521Z" fill="white"/>
                                                        <path d="M19.6285 7.53252H16.1609C15.8034 7.53252 15.5134 7.82236 15.5134 8.17998C15.5134 8.5376 15.8034 8.82744 16.1609 8.82744H19.6285C19.986 8.82744 20.276 8.5376 20.276 8.17998C20.276 7.82236 19.9861 7.53252 19.6285 7.53252Z" fill="white"/>
                                                        <path d="M19.6285 9.95982H16.1609C15.8034 9.95982 15.5134 10.2497 15.5134 10.6073C15.5134 10.9647 15.8034 11.2547 16.1609 11.2547H19.6285C19.986 11.2547 20.276 10.9647 20.276 10.6073C20.276 10.2497 19.9861 9.95982 19.6285 9.95982Z" fill="white"/>
                                                        </g>
                                                        <defs>
                                                        <clipPath id="clip0_27_598">
                                                        <rect width="22.1" height="22.1" fill="white" transform="translate(0.949951 0.949997)"/>
                                                        </clipPath>
                                                        </defs>
                                                    </svg>                                                                                                                                                                                                                             
                                                </div>
                                                <div class="staff-info">
                                                    <p class="mb-1 dashboard-link">{{__('Done Training')}}</p>
                                                    <h4 class="mb-0">{{$doneTraining}}</h4>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-6 mb-4">
                        <div class="card h-100 mb-0">
                            <div class="card-header">

                                <h5>{{__('Announcement List')}}</h5>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    @if(count($announcements) > 0)
                                        <table class="table align-items-center">
                                            <thead>
                                            <tr>
                                                <th>{{__('Title')}}</th>
                                                <th>{{__('Start Date')}}</th>
                                                <th>{{__('End Date')}}</th>

                                            </tr>
                                            </thead>
                                            <tbody class="list">
                                            @foreach($announcements as $announcement)
                                                <tr>
                                                    <td>{{ $announcement->title }}</td>
                                                    <td>{{ \Auth::user()->dateFormat($announcement->start_date) }}</td>
                                                    <td>{{ \Auth::user()->dateFormat($announcement->end_date) }}</td>

                                                </tr>
                                            @endforeach
                                            </tbody>
                                        </table>
                                    @else
                                        <div class="p-2">
                                            {{__('No accouncement present yet.')}}
                                        </div>
                                    @endif
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-6 mb-4">
                        <div class="card h-100 mb-0">
                            <div class="card-header">
                                <h5>{{__('Meeting schedule')}}</h5>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    @if(count($meetings) > 0)
                                        <table class="table align-items-center">
                                            <thead>
                                            <tr>
                                                <th>{{__('Title')}}</th>
                                                <th>{{__('Date')}}</th>
                                                <th>{{__('Time')}}</th>
                                            </tr>
                                            </thead>
                                            <tbody class="list">
                                            @foreach($meetings as $meeting)
                                                <tr>
                                                    <td>{{ $meeting->title }}</td>
                                                    <td>{{ \Auth::user()->dateFormat($meeting->date) }}</td>
                                                    <td>{{  \Auth::user()->timeFormat($meeting->time) }}</td>
                                                </tr>
                                            @endforeach
                                            </tbody>
                                        </table>
                                    @else
                                        <div class="p-2">
                                            {{__('No meeting scheduled yet.')}}
                                        </div>
                                    @endif
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    @endif
@endsection


