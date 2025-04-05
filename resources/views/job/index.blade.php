@extends('layouts.admin')
@section('page-title')
    {{__('Manage Job')}}
@endsection
@section('breadcrumb')
    <li class="breadcrumb-item"><a href="{{route('dashboard')}}">{{__('Dashboard')}}</a></li>
    <li class="breadcrumb-item">{{__('Job')}}</li>
@endsection
@push('script-page')


    <script>
        function copyToClipboard(element) {

            var copyText = element.id;
            navigator.clipboard.writeText(copyText);
            // document.addEventListener('copy', function (e) {
            //     e.clipboardData.setData('text/plain', copyText);
            //     e.preventDefault();
            // }, true);
            //
            // document.execCommand('copy');
            show_toastr('success', 'Url copied to clipboard', 'success');
        }
    </script>


@endpush


@section('action-btn')
    <div class="float-end">
        @can('create job')
            <a href="{{ route('job.create') }}" class="btn btn-sm btn-primary"  data-bs-toggle="tooltip" title="{{__('Create')}}" data-title="{{__('Create New Job')}}">
                <i class="ti ti-plus"></i>
            </a>
        @endcan
    </div>
@endsection

@section('content')
    <div class="row mb-4 gy-4">
        <div class="col-xl-4 col-sm-6 col-12 job-info-card">
            <div class="job-card-inner d-flex align-items-center gap-3">
                <div class="job-icon">
                    <div class="job-icon-inner">
                        <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M13.7617 17.5703C14.1823 17.5703 14.5234 17.2293 14.5234 16.8086V13.7617C14.5234 13.3411 14.1825 13 13.7617 13H12.2383C11.8177 13 11.4766 13.3409 11.4766 13.7617V16.8086C11.4766 17.2292 11.8176 17.5703 12.2383 17.5703H13.7617Z" fill="white"/>
                            <path d="M23.7148 4.62109H18.332C18.332 4.46768 18.332 3.70596 18.332 3.85938C18.332 2.59929 17.307 1.57422 16.0469 1.57422C15.8815 1.57422 9.72689 1.57422 9.95312 1.57422C8.69304 1.57422 7.66797 2.59924 7.66797 3.85938C7.66797 4.01279 7.66797 4.7745 7.66797 4.62109H2.28516C1.02507 4.62109 0 5.64611 0 6.90625C0 8.39805 0.182813 9.7045 0.545797 10.8179C0.908781 11.9313 1.45199 12.8516 2.17283 13.5713C3.38584 14.7827 4.93091 15.2852 6.6658 15.2852H9.95312C9.95312 15.1317 9.95312 13.6083 9.95312 13.7617C9.95312 12.5016 10.9781 11.4766 12.2383 11.4766C12.3917 11.4766 13.9151 11.4766 13.7617 11.4766C15.0218 11.4766 16.0469 12.5016 16.0469 13.7617C16.0469 13.9151 16.0469 15.4386 16.0469 15.2852H17.7394C19.2557 15.2221 21.736 15.6832 23.8264 13.6025C24.5476 12.8847 25.091 11.9627 25.4541 10.8441C25.8172 9.72547 26 8.41034 26 6.90625C26 5.64616 24.975 4.62109 23.7148 4.62109ZM9.19141 3.85938C9.19141 3.43911 9.53281 3.09766 9.95312 3.09766H16.0469C16.4671 3.09766 16.8086 3.43906 16.8086 3.85938C16.8086 4.01279 16.8086 4.7745 16.8086 4.62109H9.19141C9.19141 4.46768 9.19141 3.70596 9.19141 3.85938Z" fill="white"/>
                            <path d="M17.7602 16.8087H16.0469C16.0469 18.0688 15.0219 19.0938 13.7617 19.0938C13.6083 19.0938 12.0849 19.0938 12.2383 19.0938C10.9782 19.0938 9.95312 18.0688 9.95312 16.8087C9.7791 16.8087 6.48436 16.8087 6.67245 16.8087C4.49983 16.8087 2.59152 16.142 1.09642 14.6493C0.67407 14.2276 0.313422 13.7529 0 13.2451V23.6642C0 24.0852 0.340691 24.4259 0.761719 24.4259H25.2383C25.6593 24.4259 26 24.0852 26 23.6642V13.2764C25.6824 13.7919 25.3202 14.2656 24.9013 14.6827C22.4978 17.0737 19.9381 16.7058 17.7602 16.8087Z" fill="white"/>
                        </svg>
                    </div>
                </div>
                <div class="job-content flex-1">
                    <span class="text-muted text-sm d-block mb-1">{{__('Total')}}</span>
                    <h2 class="h5 mb-0">{{__('Jobs')}}</h2>
                </div>
                <h3 class="mb-0">{{$data['total']}}</h3>
            </div>
        </div>
        <div class="col-xl-4 col-sm-6 col-12 job-info-card">
            <div class="job-card-inner d-flex align-items-center gap-3">
                <div class="job-icon">
                    <div class="job-icon-inner">
                        <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M13.7617 17.5703C14.1823 17.5703 14.5234 17.2293 14.5234 16.8086V13.7617C14.5234 13.3411 14.1825 13 13.7617 13H12.2383C11.8177 13 11.4766 13.3409 11.4766 13.7617V16.8086C11.4766 17.2292 11.8176 17.5703 12.2383 17.5703H13.7617Z" fill="white"/>
                            <path d="M23.7148 4.62109H18.332C18.332 4.46768 18.332 3.70596 18.332 3.85938C18.332 2.59929 17.307 1.57422 16.0469 1.57422C15.8815 1.57422 9.72689 1.57422 9.95312 1.57422C8.69304 1.57422 7.66797 2.59924 7.66797 3.85938C7.66797 4.01279 7.66797 4.7745 7.66797 4.62109H2.28516C1.02507 4.62109 0 5.64611 0 6.90625C0 8.39805 0.182813 9.7045 0.545797 10.8179C0.908781 11.9313 1.45199 12.8516 2.17283 13.5713C3.38584 14.7827 4.93091 15.2852 6.6658 15.2852H9.95312C9.95312 15.1317 9.95312 13.6083 9.95312 13.7617C9.95312 12.5016 10.9781 11.4766 12.2383 11.4766C12.3917 11.4766 13.9151 11.4766 13.7617 11.4766C15.0218 11.4766 16.0469 12.5016 16.0469 13.7617C16.0469 13.9151 16.0469 15.4386 16.0469 15.2852H17.7394C19.2557 15.2221 21.736 15.6832 23.8264 13.6025C24.5476 12.8847 25.091 11.9627 25.4541 10.8441C25.8172 9.72547 26 8.41034 26 6.90625C26 5.64616 24.975 4.62109 23.7148 4.62109ZM9.19141 3.85938C9.19141 3.43911 9.53281 3.09766 9.95312 3.09766H16.0469C16.4671 3.09766 16.8086 3.43906 16.8086 3.85938C16.8086 4.01279 16.8086 4.7745 16.8086 4.62109H9.19141C9.19141 4.46768 9.19141 3.70596 9.19141 3.85938Z" fill="white"/>
                            <path d="M17.7602 16.8087H16.0469C16.0469 18.0688 15.0219 19.0938 13.7617 19.0938C13.6083 19.0938 12.0849 19.0938 12.2383 19.0938C10.9782 19.0938 9.95312 18.0688 9.95312 16.8087C9.7791 16.8087 6.48436 16.8087 6.67245 16.8087C4.49983 16.8087 2.59152 16.142 1.09642 14.6493C0.67407 14.2276 0.313422 13.7529 0 13.2451V23.6642C0 24.0852 0.340691 24.4259 0.761719 24.4259H25.2383C25.6593 24.4259 26 24.0852 26 23.6642V13.2764C25.6824 13.7919 25.3202 14.2656 24.9013 14.6827C22.4978 17.0737 19.9381 16.7058 17.7602 16.8087Z" fill="white"/>
                        </svg>
                    </div>
                </div>
                <div class="job-content flex-1">
                    <span class="text-muted text-sm d-block mb-1">{{__('Active')}}</span>
                    <h2 class="h5 mb-0">{{__('Jobs')}}</h2>
                </div>
                <h3 class="mb-0">{{$data['active']}}</h3>
            </div>
        </div>
        <div class="col-xl-4 col-sm-6 col-12 job-info-card">
            <div class="job-card-inner d-flex align-items-center gap-3">
                <div class="job-icon">
                    <div class="job-icon-inner">
                        <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M13.7617 17.5703C14.1823 17.5703 14.5234 17.2293 14.5234 16.8086V13.7617C14.5234 13.3411 14.1825 13 13.7617 13H12.2383C11.8177 13 11.4766 13.3409 11.4766 13.7617V16.8086C11.4766 17.2292 11.8176 17.5703 12.2383 17.5703H13.7617Z" fill="white"/>
                            <path d="M23.7148 4.62109H18.332C18.332 4.46768 18.332 3.70596 18.332 3.85938C18.332 2.59929 17.307 1.57422 16.0469 1.57422C15.8815 1.57422 9.72689 1.57422 9.95312 1.57422C8.69304 1.57422 7.66797 2.59924 7.66797 3.85938C7.66797 4.01279 7.66797 4.7745 7.66797 4.62109H2.28516C1.02507 4.62109 0 5.64611 0 6.90625C0 8.39805 0.182813 9.7045 0.545797 10.8179C0.908781 11.9313 1.45199 12.8516 2.17283 13.5713C3.38584 14.7827 4.93091 15.2852 6.6658 15.2852H9.95312C9.95312 15.1317 9.95312 13.6083 9.95312 13.7617C9.95312 12.5016 10.9781 11.4766 12.2383 11.4766C12.3917 11.4766 13.9151 11.4766 13.7617 11.4766C15.0218 11.4766 16.0469 12.5016 16.0469 13.7617C16.0469 13.9151 16.0469 15.4386 16.0469 15.2852H17.7394C19.2557 15.2221 21.736 15.6832 23.8264 13.6025C24.5476 12.8847 25.091 11.9627 25.4541 10.8441C25.8172 9.72547 26 8.41034 26 6.90625C26 5.64616 24.975 4.62109 23.7148 4.62109ZM9.19141 3.85938C9.19141 3.43911 9.53281 3.09766 9.95312 3.09766H16.0469C16.4671 3.09766 16.8086 3.43906 16.8086 3.85938C16.8086 4.01279 16.8086 4.7745 16.8086 4.62109H9.19141C9.19141 4.46768 9.19141 3.70596 9.19141 3.85938Z" fill="white"/>
                            <path d="M17.7602 16.8087H16.0469C16.0469 18.0688 15.0219 19.0938 13.7617 19.0938C13.6083 19.0938 12.0849 19.0938 12.2383 19.0938C10.9782 19.0938 9.95312 18.0688 9.95312 16.8087C9.7791 16.8087 6.48436 16.8087 6.67245 16.8087C4.49983 16.8087 2.59152 16.142 1.09642 14.6493C0.67407 14.2276 0.313422 13.7529 0 13.2451V23.6642C0 24.0852 0.340691 24.4259 0.761719 24.4259H25.2383C25.6593 24.4259 26 24.0852 26 23.6642V13.2764C25.6824 13.7919 25.3202 14.2656 24.9013 14.6827C22.4978 17.0737 19.9381 16.7058 17.7602 16.8087Z" fill="white"/>
                        </svg>
                    </div>
                </div>
                <div class="job-content flex-1">
                    <span class="text-muted text-sm d-block mb-1">{{__('Inactive')}}</span>
                    <h2 class="h5 mb-0">{{__('Jobs')}}</h2>
                </div>
                <h3 class="mb-0">{{$data['in_active']}}</h3>
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
                                <th>{{__('Branch')}}</th>
                                <th>{{__('Title')}}</th>
                                <th>{{__('Start Date')}}</th>
                                <th>{{__('End Date')}}</th>
                                <th>{{__('Status')}}</th>
                                <th>{{__('Created At')}}</th>
                                @if( Gate::check('edit job') ||Gate::check('delete job') ||Gate::check('show job'))
                                    <th width="200px">{{__('Action')}}</th>
                                @endif
                            </tr>
                            </thead>
                            <tbody class="font-style">
                            @foreach ($jobs as $job)
                                <tr>
                                    <td>{{ !empty($job->branches)?$job->branches->name:__('All') }}</td>
                                    <td>{{$job->title}}</td>
                                    <td>{{\Auth::user()->dateFormat($job->start_date)}}</td>
                                    <td>{{\Auth::user()->dateFormat($job->end_date)}}</td>
                                    <td>
                                        @if($job->status=='active')
                                            <span class="status_badge badge bg-primary p-2 px-3 rounded">{{App\Models\Job::$status[$job->status]}}</span>
                                        @else
                                            <span class="status_badge badge bg-danger p-2 px-3 rounded">{{App\Models\Job::$status[$job->status]}}</span>
                                        @endif
                                    </td>
                                    <td>{{ \Auth::user()->dateFormat($job->created_at) }}</td>
                                    @if( Gate::check('edit job') ||Gate::check('delete job') || Gate::check('show job'))
                                        <td>

                                        @if($job->status!='in_active')
{{--                                            <div class="action-btn bg-warning ms-2">--}}
{{--                                                <a href="{{ route('job.requirement',[$job->code,!empty($job)?$job->createdBy->lang:'en']) }}" class="mx-3 btn btn-sm align-items-center " onclick="copyToClipboard(this)" data-bs-toggle="tooltip" data-original-title="{{__('Click to copy')}}">--}}
{{--                                                    <i class="ti ti-link text-white"></i></a>--}}

{{--                                                <a href="#" id="{{ route('invoice.link.copy',[$invoiceID]) }}" class="mx-3 btn btn-sm align-items-center"   onclick="copyToClipboard(this)" data-bs-toggle="tooltip" data-original-title="{{__('Click to copy')}}"><i class="ti ti-link text-white"></i></a>--}}

{{--                                            </div>--}}

                                                <div class="action-btn me-2">
                                                    <a href="#" id="{{ route('job.requirement',[$job->code,!empty($job)?$job->createdBy->lang:'en']) }}" class="mx-3 btn btn-sm align-items-center bg-secondary"  onclick="copyToClipboard(this)" data-bs-toggle="tooltip" title="{{__('Copy')}}" data-original-title="{{__('Click to copy')}}"><i class="ti ti-link text-white"></i></a>
                                                </div>


                                            @endif
                                            @can('show job')
                                            <div class="action-btn me-2">
                                                <a href="{{ route('job.show',$job->id) }}" data-title="{{__('Job Detail')}}" title="{{__('View')}}"  class="mx-3 btn btn-sm align-items-center bg-warning" data-bs-toggle="tooltip" data-original-title="{{__('View Detail')}}">
                                                    <i class="ti ti-eye text-white"></i></a>
                                            </div>
                                                @endcan
                                            @can('edit job')
                                            <div class="action-btn me-2">
                                                <a href="{{ route('job.edit',$job->id) }}" data-title="{{__('Edit Job')}}" class="mx-3 btn btn-sm align-items-center bg-info" data-bs-toggle="tooltip" title="{{__('Edit')}}" data-original-title="{{__('Edit')}}">
                                                    <i class="ti ti-pencil text-white"></i></a>
                                            </div>
                                                @endcan
                                            @can('delete job')
                                            <div class="action-btn ">
                                            {!! Form::open(['method' => 'DELETE', 'route' => ['job.destroy', $job->id],'id'=>'delete-form-'.$job->id]) !!}

                                                <a href="#" class="mx-3 btn btn-sm align-items-center bs-pass-para bg-danger" data-bs-toggle="tooltip" title="{{__('Delete')}}" data-original-title="{{__('Delete')}}" data-confirm="{{__('Are You Sure?').'|'.__('This action can not be undone. Do you want to continue?')}}" data-confirm-yes="document.getElementById('delete-form-{{$job->id}}').submit();">
                                                    <i class="ti ti-trash text-white"></i></a>
                                                {!! Form::close() !!}
                                                </div>
                                            @endcan
                                        </td>
                                    @endif
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
