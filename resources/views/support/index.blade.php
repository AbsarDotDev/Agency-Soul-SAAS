@extends('layouts.admin')
@push('script-page')
@endpush
@section('page-title')
    {{__('Support')}}
@endsection
@section('title')
    <div class="d-inline-block">
        <h5 class="h4 d-inline-block font-weight-400 mb-0 ">{{__('Support')}}</h5>
    </div>
@endsection
@section('breadcrumb')
    <li class="breadcrumb-item"><a href="{{route('dashboard')}}">{{__('Dashboard')}}</a></li>
    <li class="breadcrumb-item">{{__('Support')}}</li>
@endsection

@section('action-btn')
    <div class="float-end">
        <a href="{{ route('support.grid') }}" class="btn btn-sm btn-primary-subtle me-1" data-bs-toggle="tooltip" title="{{__('Grid View')}}">
            <i class="ti ti-layout-grid text-white"></i>
        </a>
       <a href="#" data-size="lg" data-url="{{ route('support.create') }}" data-ajax-popup="true" data-bs-toggle="tooltip" title="{{__('Create')}}" data-title="{{__('Create Support')}}" class="btn btn-sm btn-primary">
            <i class="ti ti-plus"></i>
        </a>
    </div>
@endsection

@section('content')
    <div class="row mb-4 gy-4">
        <div class="col-xxl-3 col-xl-4 col-sm-6 col-12 support-ticket-card">
            <div class="support-card-inner d-flex align-items-center gap-3">
                <div class="support-icon">
                    <div class="icon-inner">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M23.0002 12.9999C23.2502 12.9999 23.5003 12.7498 23.5003 12.4998V9.49976C23.5003 8.64968 22.8504 7.99976 22.0003 7.99976H18V21.9999H21.9998C22.8499 21.9999 23.4998 21.35 23.4998 20.4999V17.4999C23.4996 17.3673 23.4468 17.2403 23.3531 17.1465C23.2593 17.0528 23.1323 17 22.9997 16.9998C22.4746 16.9916 21.9738 16.7773 21.6054 16.4032C21.237 16.029 21.0305 15.5249 21.0305 14.9998C21.0305 14.4747 21.237 13.9707 21.6054 13.5965C21.9738 13.2223 22.4746 13.008 22.9997 12.9999H23.0002Z" fill="white"/>
                            <path d="M22.6496 6.50021C22.8997 6.40037 23.0494 6.10037 22.9496 5.85029L21.9493 3.00005C21.6992 2.19989 20.7973 1.80005 20.0494 2.05013L5.84961 6.94997H21.8495C22.0866 6.75262 22.3578 6.60021 22.6496 6.50021Z" fill="white"/>
                            <path d="M0.5 9.50024V12.5002C0.5 12.8002 0.70016 13.0004 1.00016 13.0004C1.2654 12.9963 1.52881 13.045 1.77505 13.1437C2.02129 13.2423 2.24545 13.389 2.43447 13.5751C2.6235 13.7612 2.77361 13.9831 2.87607 14.2277C2.97854 14.4724 3.0313 14.7351 3.0313 15.0003C3.0313 15.2656 2.97854 15.5282 2.87607 15.7729C2.77361 16.0176 2.6235 16.2394 2.43447 16.4256C2.24545 16.6117 2.02129 16.7583 1.77505 16.857C1.52881 16.9557 1.2654 17.0044 1.00016 17.0002C0.70016 17.0002 0.5 17.2004 0.5 17.5004V20.5004C0.5 21.3505 1.14992 22.0004 2 22.0004H17V8.00024H2C1.15184 8.00024 0.5 8.65016 0.5 9.50024ZM6.5 12.0001H9.5C9.8 12.0001 10.0002 12.2002 10.0002 12.5002C10.0002 12.8002 9.8 13.0004 9.5 13.0004H6.5C6.2 13.0004 5.99984 12.8002 5.99984 12.5002C5.99984 12.2002 6.2 12.0001 6.5 12.0001ZM6.5 14.4999H12.9997C13.2997 14.4999 13.4998 14.7001 13.4998 15.0001C13.4998 15.3001 13.2997 15.5002 12.9997 15.5002H6.5C6.2 15.5002 5.99984 15.3001 5.99984 15.0001C5.99984 14.7001 6.2 14.4999 6.5 14.4999ZM6.5 16.9998H12.9997C13.2997 16.9998 13.4998 17.1999 13.4998 17.4999C13.4998 17.7999 13.2997 18.0001 12.9997 18.0001H6.5C6.2 18.0001 5.99984 17.7999 5.99984 17.4999C5.99984 17.1999 6.2 17.0002 6.5 17.0002V16.9998Z" fill="white"/>
                        </svg>                            
                    </div>
                </div>
                <div class="support-content flex-1">
                    <span class="text-muted text-sm d-block mb-1">{{__('Total')}}</span>
                    <h2 class="h5 mb-0">{{__('Ticket')}}</h2>
                </div>
                <h3 class="mb-0">{{ $countTicket }}</h3>
            </div>
        </div>
        <div class="col-xxl-3 col-xl-4 col-sm-6 col-12 support-ticket-card">
            <div class="support-card-inner d-flex align-items-center gap-3">
                <div class="support-icon">
                    <div class="icon-inner">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M23.0002 12.9999C23.2502 12.9999 23.5003 12.7498 23.5003 12.4998V9.49976C23.5003 8.64968 22.8504 7.99976 22.0003 7.99976H18V21.9999H21.9998C22.8499 21.9999 23.4998 21.35 23.4998 20.4999V17.4999C23.4996 17.3673 23.4468 17.2403 23.3531 17.1465C23.2593 17.0528 23.1323 17 22.9997 16.9998C22.4746 16.9916 21.9738 16.7773 21.6054 16.4032C21.237 16.029 21.0305 15.5249 21.0305 14.9998C21.0305 14.4747 21.237 13.9707 21.6054 13.5965C21.9738 13.2223 22.4746 13.008 22.9997 12.9999H23.0002Z" fill="white"/>
                            <path d="M22.6496 6.50021C22.8997 6.40037 23.0494 6.10037 22.9496 5.85029L21.9493 3.00005C21.6992 2.19989 20.7973 1.80005 20.0494 2.05013L5.84961 6.94997H21.8495C22.0866 6.75262 22.3578 6.60021 22.6496 6.50021Z" fill="white"/>
                            <path d="M0.5 9.50024V12.5002C0.5 12.8002 0.70016 13.0004 1.00016 13.0004C1.2654 12.9963 1.52881 13.045 1.77505 13.1437C2.02129 13.2423 2.24545 13.389 2.43447 13.5751C2.6235 13.7612 2.77361 13.9831 2.87607 14.2277C2.97854 14.4724 3.0313 14.7351 3.0313 15.0003C3.0313 15.2656 2.97854 15.5282 2.87607 15.7729C2.77361 16.0176 2.6235 16.2394 2.43447 16.4256C2.24545 16.6117 2.02129 16.7583 1.77505 16.857C1.52881 16.9557 1.2654 17.0044 1.00016 17.0002C0.70016 17.0002 0.5 17.2004 0.5 17.5004V20.5004C0.5 21.3505 1.14992 22.0004 2 22.0004H17V8.00024H2C1.15184 8.00024 0.5 8.65016 0.5 9.50024ZM6.5 12.0001H9.5C9.8 12.0001 10.0002 12.2002 10.0002 12.5002C10.0002 12.8002 9.8 13.0004 9.5 13.0004H6.5C6.2 13.0004 5.99984 12.8002 5.99984 12.5002C5.99984 12.2002 6.2 12.0001 6.5 12.0001ZM6.5 14.4999H12.9997C13.2997 14.4999 13.4998 14.7001 13.4998 15.0001C13.4998 15.3001 13.2997 15.5002 12.9997 15.5002H6.5C6.2 15.5002 5.99984 15.3001 5.99984 15.0001C5.99984 14.7001 6.2 14.4999 6.5 14.4999ZM6.5 16.9998H12.9997C13.2997 16.9998 13.4998 17.1999 13.4998 17.4999C13.4998 17.7999 13.2997 18.0001 12.9997 18.0001H6.5C6.2 18.0001 5.99984 17.7999 5.99984 17.4999C5.99984 17.1999 6.2 17.0002 6.5 17.0002V16.9998Z" fill="white"/>
                        </svg>                            
                    </div>
                </div>
                <div class="support-content flex-1">
                    <span class="text-muted text-sm d-block mb-1">{{__('Open')}}</span>
                    <h2 class="h5 mb-0">{{__('Ticket')}}</h2>
                </div>
                <h3 class="mb-0">{{ $countOpenTicket }}</h3>
            </div>
        </div>
        <div class="col-xxl-3 col-xl-4 col-sm-6 col-12 support-ticket-card">
            <div class="support-card-inner d-flex align-items-center gap-3">
                <div class="support-icon">
                    <div class="icon-inner">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M23.0002 12.9999C23.2502 12.9999 23.5003 12.7498 23.5003 12.4998V9.49976C23.5003 8.64968 22.8504 7.99976 22.0003 7.99976H18V21.9999H21.9998C22.8499 21.9999 23.4998 21.35 23.4998 20.4999V17.4999C23.4996 17.3673 23.4468 17.2403 23.3531 17.1465C23.2593 17.0528 23.1323 17 22.9997 16.9998C22.4746 16.9916 21.9738 16.7773 21.6054 16.4032C21.237 16.029 21.0305 15.5249 21.0305 14.9998C21.0305 14.4747 21.237 13.9707 21.6054 13.5965C21.9738 13.2223 22.4746 13.008 22.9997 12.9999H23.0002Z" fill="white"/>
                            <path d="M22.6496 6.50021C22.8997 6.40037 23.0494 6.10037 22.9496 5.85029L21.9493 3.00005C21.6992 2.19989 20.7973 1.80005 20.0494 2.05013L5.84961 6.94997H21.8495C22.0866 6.75262 22.3578 6.60021 22.6496 6.50021Z" fill="white"/>
                            <path d="M0.5 9.50024V12.5002C0.5 12.8002 0.70016 13.0004 1.00016 13.0004C1.2654 12.9963 1.52881 13.045 1.77505 13.1437C2.02129 13.2423 2.24545 13.389 2.43447 13.5751C2.6235 13.7612 2.77361 13.9831 2.87607 14.2277C2.97854 14.4724 3.0313 14.7351 3.0313 15.0003C3.0313 15.2656 2.97854 15.5282 2.87607 15.7729C2.77361 16.0176 2.6235 16.2394 2.43447 16.4256C2.24545 16.6117 2.02129 16.7583 1.77505 16.857C1.52881 16.9557 1.2654 17.0044 1.00016 17.0002C0.70016 17.0002 0.5 17.2004 0.5 17.5004V20.5004C0.5 21.3505 1.14992 22.0004 2 22.0004H17V8.00024H2C1.15184 8.00024 0.5 8.65016 0.5 9.50024ZM6.5 12.0001H9.5C9.8 12.0001 10.0002 12.2002 10.0002 12.5002C10.0002 12.8002 9.8 13.0004 9.5 13.0004H6.5C6.2 13.0004 5.99984 12.8002 5.99984 12.5002C5.99984 12.2002 6.2 12.0001 6.5 12.0001ZM6.5 14.4999H12.9997C13.2997 14.4999 13.4998 14.7001 13.4998 15.0001C13.4998 15.3001 13.2997 15.5002 12.9997 15.5002H6.5C6.2 15.5002 5.99984 15.3001 5.99984 15.0001C5.99984 14.7001 6.2 14.4999 6.5 14.4999ZM6.5 16.9998H12.9997C13.2997 16.9998 13.4998 17.1999 13.4998 17.4999C13.4998 17.7999 13.2997 18.0001 12.9997 18.0001H6.5C6.2 18.0001 5.99984 17.7999 5.99984 17.4999C5.99984 17.1999 6.2 17.0002 6.5 17.0002V16.9998Z" fill="white"/>
                        </svg>                            
                    </div>
                </div>
                <div class="support-content flex-1">
                    <span class="text-muted text-sm d-block mb-1">{{__('On Hold')}}</span>
                    <h2 class="h5 mb-0">{{__('Ticket')}}</h2>
                </div>
                <h3 class="mb-0">{{ $countonholdTicket }}</h3>
            </div>
        </div>
        <div class="col-xxl-3 col-xl-4 col-sm-6 col-12 support-ticket-card">
            <div class="support-card-inner d-flex align-items-center gap-3">
                <div class="support-icon">
                    <div class="icon-inner">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M23.0002 12.9999C23.2502 12.9999 23.5003 12.7498 23.5003 12.4998V9.49976C23.5003 8.64968 22.8504 7.99976 22.0003 7.99976H18V21.9999H21.9998C22.8499 21.9999 23.4998 21.35 23.4998 20.4999V17.4999C23.4996 17.3673 23.4468 17.2403 23.3531 17.1465C23.2593 17.0528 23.1323 17 22.9997 16.9998C22.4746 16.9916 21.9738 16.7773 21.6054 16.4032C21.237 16.029 21.0305 15.5249 21.0305 14.9998C21.0305 14.4747 21.237 13.9707 21.6054 13.5965C21.9738 13.2223 22.4746 13.008 22.9997 12.9999H23.0002Z" fill="white"/>
                            <path d="M22.6496 6.50021C22.8997 6.40037 23.0494 6.10037 22.9496 5.85029L21.9493 3.00005C21.6992 2.19989 20.7973 1.80005 20.0494 2.05013L5.84961 6.94997H21.8495C22.0866 6.75262 22.3578 6.60021 22.6496 6.50021Z" fill="white"/>
                            <path d="M0.5 9.50024V12.5002C0.5 12.8002 0.70016 13.0004 1.00016 13.0004C1.2654 12.9963 1.52881 13.045 1.77505 13.1437C2.02129 13.2423 2.24545 13.389 2.43447 13.5751C2.6235 13.7612 2.77361 13.9831 2.87607 14.2277C2.97854 14.4724 3.0313 14.7351 3.0313 15.0003C3.0313 15.2656 2.97854 15.5282 2.87607 15.7729C2.77361 16.0176 2.6235 16.2394 2.43447 16.4256C2.24545 16.6117 2.02129 16.7583 1.77505 16.857C1.52881 16.9557 1.2654 17.0044 1.00016 17.0002C0.70016 17.0002 0.5 17.2004 0.5 17.5004V20.5004C0.5 21.3505 1.14992 22.0004 2 22.0004H17V8.00024H2C1.15184 8.00024 0.5 8.65016 0.5 9.50024ZM6.5 12.0001H9.5C9.8 12.0001 10.0002 12.2002 10.0002 12.5002C10.0002 12.8002 9.8 13.0004 9.5 13.0004H6.5C6.2 13.0004 5.99984 12.8002 5.99984 12.5002C5.99984 12.2002 6.2 12.0001 6.5 12.0001ZM6.5 14.4999H12.9997C13.2997 14.4999 13.4998 14.7001 13.4998 15.0001C13.4998 15.3001 13.2997 15.5002 12.9997 15.5002H6.5C6.2 15.5002 5.99984 15.3001 5.99984 15.0001C5.99984 14.7001 6.2 14.4999 6.5 14.4999ZM6.5 16.9998H12.9997C13.2997 16.9998 13.4998 17.1999 13.4998 17.4999C13.4998 17.7999 13.2997 18.0001 12.9997 18.0001H6.5C6.2 18.0001 5.99984 17.7999 5.99984 17.4999C5.99984 17.1999 6.2 17.0002 6.5 17.0002V16.9998Z" fill="white"/>
                        </svg>                            
                    </div>
                </div>
                <div class="support-content flex-1">
                    <span class="text-muted text-sm d-block mb-1">{{__('Close')}}</span>
                    <h2 class="h5 mb-0">{{__('Ticket')}}</h2>
                </div>
                <h3 class="mb-0">{{ $countCloseTicket }}</h3>
            </div>
        </div>
    </div>

<div class="row">
    <div class="col-md-12">
        <div class="card">
            <div class="card-body table-border-style">
                <div class="table-responsive">
                    <table class="table datatable">
                        <thead>
                        <tr>
                            <th scope="col">{{__('Created By')}}</th>
                            <th scope="col">{{__('Ticket')}}</th>
                            <th scope="col">{{__('Code')}}</th>
                            <th scope="col">{{__('Attachment')}}</th>
                            <th scope="col">{{__('Assign User')}}</th>
                            <th scope="col">{{__('Status')}}</th>
                            <th scope="col">{{__('Created At')}}</th>
                            <th scope="col" >{{__('Action')}}</th>
                        </tr>
                        </thead>
                        <tbody class="list">
                            @php
                                $supportpath=\App\Models\Utility::get_file('uploads/supports');
                            @endphp
                            @foreach($supports as $support)
                                <tr>
                                    <td scope="row">
                                        <div class="media align-items-center">
                                            <div>
                                                <div class="avatar-parent-child">
                                                    <img alt="" class="avatar rounded border-2 border border-primary avatar-sm me-1" @if(!empty($support->createdBy) && !empty($support->createdBy->avatar) && file_exists('storage/uploads/avatar/'.$support->createdBy->avatar)) src="{{asset(Storage::url('uploads/avatar')).'/'.$support->createdBy->avatar}}" @else  src="{{asset(Storage::url('uploads/avatar')).'/avatar.png'}}" @endif>
                                                    @if($support->replyUnread()>0)
                                                        <span class="avatar-child avatar-badge bg-success"></span>
                                                    @endif
                                                </div>
                                            </div>
                                            <div class="media-body">
                                                {{!empty($support->createdBy)?$support->createdBy->name:''}}
                                            </div>
                                        </div>
                                    </td>
                                    <td scope="row">
                                        <div class="media align-items-center">
                                            <div class="media-body">
                                                <a href="{{ route('support.reply',\Crypt::encrypt($support->id)) }}" class="name h6 mb-2 d-block text-sm">{{$support->subject}}</a>
                                                @if($support->priority == 0)
                                                    <span data-toggle="tooltip" data-title="{{__('Priority')}}" class="text-capitalize status_badge badge bg-primary p-2 px-3 rounded">   {{ __(\App\Models\Support::$priority[$support->priority]) }}</span>
                                                @elseif($support->priority == 1)
                                                    <span data-toggle="tooltip" data-title="{{__('Priority')}}" class="text-capitalize status_badge badge bg-info p-2 px-3 rounded">   {{ __(\App\Models\Support::$priority[$support->priority]) }}</span>
                                                @elseif($support->priority == 2)
                                                    <span data-toggle="tooltip" data-title="{{__('Priority')}}" class="text-capitalize status_badge badge bg-warning p-2 px-3 rounded">   {{ __(\App\Models\Support::$priority[$support->priority]) }}</span>
                                                @elseif($support->priority == 3)
                                                    <span data-toggle="tooltip" data-title="{{__('Priority')}}" class="text-capitalize status_badge badge bg-danger p-2 px-3 rounded">   {{ __(\App\Models\Support::$priority[$support->priority]) }}</span>
                                                @endif
                                            </div>
                                        </div>
                                    </td>
                                    <td>{{$support->ticket_code}}</td>
                                    <td>
                                        @if(!empty($support->attachment))
                                            <a  class="bg-primary ms-2 btn btn-sm align-items-center" href="{{ $supportpath . '/' . $support->attachment }}" download=""  data-bs-toggle="tooltip" title="{{__('Download')}}" target="_blank">
                                                <i class="ti ti-download text-white"></i>
                                            </a>
                                            <a href="{{ $supportpath . '/' . $support->attachment }}"
                                               class=" bg-secondary ms-2 mx-3 btn btn-sm align-items-center" data-bs-toggle="tooltip" title="{{__('Preview')}}">
                                                <span class="btn-inner--icon"><i class="ti ti-crosshair text-white" ></i></span>
                                            </a>
                                        @else
                                            -
                                        @endif

                                    </td>
                                    <td>{{!empty($support->assignUser)?$support->assignUser->name:'-'}}</td>

                                    <td>
                                        @if($support->status == 'Open')
                                            <span class="status_badge text-capitalize badge bg-success p-2 px-3 rounded">{{ __(\App\Models\Support::$status[$support->status]) }}</span>
                                        @elseif($support->status == 'Close')
                                            <span class="status_badge text-capitalize badge bg-danger p-2 px-3 rounded">{{ __(\App\Models\Support::$status[$support->status]) }}</span>
                                        @elseif($support->status == 'On Hold')
                                            <span  class="status_badge text-capitalize badge bg-warning p-2 px-3 rounded">{{ __(\App\Models\Support::$status[$support->status]) }}</span>
                                        @endif
                                    </td>
                                    <td>{{\Auth::user()->dateFormat($support->created_at)}}</td>
                                    <td class="Action">
                                    <span>
                                        <div class="action-btn me-2">
                                            <a href="{{ route('support.reply',\Crypt::encrypt($support->id)) }}" data-title="{{__('Support Reply')}}" class="mx-3 btn btn-sm align-items-center bg-warning" data-bs-toggle="tooltip" title="{{__('Reply')}}" data-original-title="{{__('Reply')}}">
                                                <i class="ti ti-corner-up-left text-white"></i>
                                            </a>
                                        </div>
                                        @if(\Auth::user()->type=='company' || \Auth::user()->id==$support->ticket_created)
                                            <div class="action-btn me-2">
                                                <a href="#" data-size="lg" data-url="{{ route('support.edit',$support->id) }}" data-ajax-popup="true" data-title="{{__('Edit Support')}}" class="mx-3 btn btn-sm align-items-center bg-info" data-bs-toggle="tooltip" title="{{__('Edit')}}" data-original-title="{{__('Edit')}}">
                                                    <i class="ti ti-pencil text-white"></i>
                                                </a>
                                            </div>
                                            <div class="action-btn ">
                                                {!! Form::open(['method' => 'DELETE', 'route' => ['support.destroy', $support->id],'id'=>'delete-form-'.$support->id]) !!}
                                                    <a href="#!" class="mx-3 btn btn-sm  align-items-center bs-pass-para bg-danger" data-bs-toggle="tooltip" data-original-title="{{__('Delete')}}" data-confirm="Are You Sure?|This action can not be undone. Do you want to continue?" title="{{__('Delete')}}" data-confirm-yes="document.getElementById('delete-form-{{$support->id}}').submit();">
                                                        <i class="ti ti-trash text-white"></i>
                                                    </a>
                                                 {!! Form::close() !!}
                                            </div>

                                        @endif
                                    </span>
                                    </td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        </div>
    </div>
@endsection

