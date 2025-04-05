{{-- <div class="card sticky-top" style="top:30px">
    <div class="list-group list-group-flush" id="useradd-sidenav">
        <a href="{{route('branch.index')}}" class="list-group-item list-group-item-action border-0 {{ (request()->is('branch*') ? 'active' : '')}}">{{__('Branch')}} <div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('department.index') }}" class="list-group-item list-group-item-action border-0 {{ (request()->is('department*') ? 'active' : '')}}">{{__('Department')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('designation.index') }}" class="list-group-item list-group-item-action border-0 {{ (request()->is('designation*') ? 'active' : '')}}">{{__('Designation')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('leavetype.index') }}" class="list-group-item list-group-item-action border-0 {{ (Request::route()->getName() == 'leavetype.index' ? 'active' : '')}}">{{__('Leave Type')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('document.index') }}" class="list-group-item list-group-item-action border-0 {{ (Request::route()->getName() == 'document.index' ? 'active' : '')}}">{{__('Document Type')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('paysliptype.index') }}" class="list-group-item list-group-item-action border-0 {{ (request()->is('paysliptype*') ? 'active' : '')}}">{{__('Payslip Type')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('allowanceoption.index') }}" class="list-group-item list-group-item-action border-0 {{ (request()->is('allowanceoption*') ? 'active' : '')}}">{{__('Allowance Option')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('loanoption.index') }}" class="list-group-item list-group-item-action border-0 {{ (request()->is('loanoption*') ? 'active' : '')}}">{{__('Loan Option')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('deductionoption.index') }}" class="list-group-item list-group-item-action border-0 {{ (request()->is('deductionoption*') ? 'active' : '')}}">{{__('Deduction Option')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('goaltype.index') }}" class="list-group-item list-group-item-action border-0 {{ (request()->is('goaltype*') ? 'active' : '')}}">{{__('Goal Type')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('trainingtype.index') }}" class="list-group-item list-group-item-action border-0 {{ (request()->is('trainingtype*') ? 'active' : '')}}">{{__('Training Type')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('awardtype.index') }}" class="list-group-item list-group-item-action border-0 {{ (request()->is('awardtype*') ? 'active' : '')}}">{{__('Award Type')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('terminationtype.index') }}" class="list-group-item list-group-item-action border-0 {{ (request()->is('terminationtype*') ? 'active' : '')}}">{{__('Termination Type')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('job-category.index') }}" class="list-group-item list-group-item-action border-0 {{ (request()->is('job-category*') ? 'active' : '')}}">{{__('Job Category')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        <a href="{{ route('job-stage.index') }}" class="list-group-item list-group-item-action border-0 {{ (request()->is('job-stage*') ? 'active' : '')}}">{{__('Job Stage')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

        @can('manage performance type')
        <a href="{{ route('performanceType.index') }}" class="list-group-item list-group-item-action border-0 {{ request()->is('performanceType*') ? 'active' : '' }}">{{__('Performance Type')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>
        @endcan

        <a href="{{ route('competencies.index') }}" class="list-group-item list-group-item-action border-0 {{ request()->is('competencies*') ? 'active' : '' }}">{{__('Competencies')}}<div class="float-end"><i class="ti ti-chevron-right"></i></div></a>

    </div>
</div> --}}

<ul class="card flex-row nav nav-pills nav-fill information-tab hrm_setup_tab" id="pills-tab" role="tablist">
    @can('manage branch')
        <li class="nav-item" role="presentation">
            <a href="{{ route('branch.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ Request::route()->getName() == 'branch.index' ? 'active' : '' }}"
                    id="branch-setting-tab" data-bs-toggle="pill" data-bs-target="#branch-setting"
                    type="button">{{ __('Branch') }}</button>
            </a>
        </li>
    @endcan
    @can('manage department')
        <li class="nav-item" role="presentation">
            <a href="{{ route('department.index') }}" class="list-group-item list-group-item-action border-0 ">
                <button class="nav-link {{ Request::route()->getName() == 'department.index' ? 'active' : '' }}"
                    id="department-setting-tab" data-bs-toggle="pill" data-bs-target="#department-setting"
                    type="button">{{ __('Department') }}</button>
            </a>
        </li>
    @endcan
    @can('manage designation')
        <li class="nav-item" role="presentation">
            <a href="{{ route('designation.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('designation*') ? 'active' : '' }}" id="designation-setting-tab"
                    data-bs-toggle="pill" data-bs-target="#designation-setting"
                    type="button">{{ __('Designation') }}</button>
            </a>
        </li>
    @endcan
    @can('manage leave type')
        <li class="nav-item" role="presentation">
            <a href="{{ route('leavetype.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ Request::route()->getName() == 'leavetype.index' ? 'active' : '' }}"
                    id="leave-setting-tab" data-bs-toggle="pill" data-bs-target="#leave-setting"
                    type="button">{{ __('Leave Type') }}</button>
            </a>
        </li>
    @endcan
    @can('manage document type')
        <li class="nav-item" role="presentation">
            <a href="{{ route('document.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ Request::route()->getName() == 'document.index' ? 'active' : '' }}"
                    id="document-setting-tab" data-bs-toggle="pill" data-bs-target="#document-setting"
                    type="button">{{ __('Document Type') }}</button>
            </a>
        </li>
    @endcan
    @can('manage payslip type')
        <li class="nav-item" role="presentation">
            <a href="{{ route('paysliptype.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('paysliptype*') ? 'active' : '' }} " id="payslip-setting-tab"
                    data-bs-toggle="pill" data-bs-target="#payslip-setting"
                    type="button">{{ __('Payslip Type') }}</button>
            </a>
        </li>
    @endcan
    @can('manage allowance option')
        <li class="nav-item" role="presentation">
            <a href="{{ route('allowanceoption.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('allowanceoption*') ? 'active' : '' }} "
                    id="allowance-setting-tab" data-bs-toggle="pill" data-bs-target="#allowance-setting"
                    type="button">{{ __('Allowance Option') }}</button>
            </a>
        </li>
    @endcan
    @can('manage loan option')
        <li class="nav-item" role="presentation">
            <a href="{{ route('loanoption.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('loanoption*') ? 'active' : '' }} " id="loan-setting-tab"
                    data-bs-toggle="pill" data-bs-target="#loan-setting" type="button">{{ __('Loan Option') }}</button>
            </a>
        </li>
    @endcan
    @can('manage deduction option')
        <li class="nav-item" role="presentation">
            <a href="{{ route('deductionoption.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('deductionoption*') ? 'active' : '' }} "
                    id="deduction-setting-tab" data-bs-toggle="pill" data-bs-target="#deduction-setting"
                    type="button">{{ __('Deduction Option') }}</button>
            </a>
        </li>
    @endcan
    @can('manage goal type')
        <li class="nav-item" role="presentation">
            <a href="{{ route('goaltype.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('goaltype*') ? 'active' : '' }} " id="goal-setting-tab"
                    data-bs-toggle="pill" data-bs-target="#goal-setting" type="button">{{ __('Goal Type') }}</button>
            </a>
        </li>
    @endcan
    @can('manage training type')
        <li class="nav-item" role="presentation">
            <a href="{{ route('trainingtype.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('trainingtype*') ? 'active' : '' }} " id="training-setting-tab"
                    data-bs-toggle="pill" data-bs-target="#training-setting"
                    type="button">{{ __('Training Type') }}</button>
            </a>
        </li>
    @endcan
    @can('manage award type')
        <li class="nav-item" role="presentation">
            <a href="{{ route('awardtype.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('awardtype*') ? 'active' : '' }} " id="award-setting-tab"
                    data-bs-toggle="pill" data-bs-target="#award-setting" type="button">{{ __('Award Type') }}</button>
            </a>
        </li>
    @endcan
    @can('manage termination type')
        <li class="nav-item" role="presentation">
            <a href="{{ route('terminationtype.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('terminationtype*') ? 'active' : '' }} "
                    id="termination-setting-tab" data-bs-toggle="pill" data-bs-target="#termination-setting"
                    type="button">{{ __('Termination Type') }}</button>
            </a>
        </li>
    @endcan
    @can('manage job category')
        <li class="nav-item" role="presentation">
            <a href="{{ route('job-category.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('job-category*') ? 'active' : '' }} "
                    id="jobcategory-setting-tab" data-bs-toggle="pill" data-bs-target="#jobcategory-setting"
                    type="button">{{ __('Job Category') }}</button>
            </a>
        </li>
    @endcan
    @can('manage job stage')
        <li class="nav-item" role="presentation">
            <a href="{{ route('job-stage.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('job-stage*') ? 'active' : '' }} " id="jobstage-setting-tab"
                    data-bs-toggle="pill" data-bs-target="#jobstage-setting"
                    type="button">{{ __('Job Stage') }}</button>
            </a>
        </li>
    @endcan
    @can('manage performance type')
        <li class="nav-item" role="presentation">
            <a href="{{ route('performanceType.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('performanceType*') ? 'active' : '' }} "
                    id="performance-setting-tab" data-bs-toggle="pill" data-bs-target="#performance-setting"
                    type="button">{{ __('Performance Type') }}</button>
            </a>
        </li>
    @endcan
    @can('Manage Competencies')
        <li class="nav-item" role="presentation">
            <a href="{{ route('competencies.index') }}" class="list-group-item list-group-item-action border-0">
                <button class="nav-link {{ request()->is('competencies*') ? 'active' : '' }} "
                    id="competencies-setting-tab" data-bs-toggle="pill" data-bs-target="#competencies-setting"
                    type="button">{{ __('Competencies') }}</button>
            </a>
        </li>
    @endcan
</ul>
