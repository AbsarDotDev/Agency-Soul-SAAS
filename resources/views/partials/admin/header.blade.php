@php
    $users=\Auth::user();
    $profile=\App\Models\Utility::get_file('uploads/avatar/');
    $languages=\App\Models\Utility::languages();

    $lang = isset($users->lang)?$users->lang:'en';
    if ($lang == null) {
        $lang = 'en';
    }
    // $LangName = \App\Models\Language::where('code',$lang)->first();
    // $LangName =\App\Models\Language::languageData($lang);
    $LangName = cache()->remember('full_language_data_' . $lang, now()->addHours(24), function () use ($lang) {
    return \App\Models\Language::languageData($lang);
    });

    $setting = \App\Models\Utility::settings();

    $unseenCounter=App\Models\ChMessage::where('to_id', Auth::user()->id)->where('seen', 0)->count();
@endphp
@if (isset($setting['cust_theme_bg']) && $setting['cust_theme_bg'] == 'on')
    <header class="dash-header transprent-bg">
@else
    <header class="dash-header">
@endif
    <div class="header-wrapper">
        <div class="me-auto dash-mob-drp">
            <ul class="list-unstyled">
                <li class="dash-h-item mob-hamburger">
                    <a href="#!" class="dash-head-link" id="mobile-collapse">
                        <div class="hamburger hamburger--arrowturn">
                            <div class="hamburger-box">
                                <div class="hamburger-inner"></div>
                            </div>
                        </div>
                    </a>
                </li>

                <li class="dropdown dash-h-item drp-company">
                    <a class="dash-head-link dropdown-toggle arrow-none me-0" data-bs-toggle="dropdown" href="#" role="button" aria-haspopup="false" aria-expanded="false">
                        <span class="theme-avtar">
                             <img src="{{ !empty(\Auth::user()->avatar) ? $profile . \Auth::user()->avatar :  $profile.'avatar.png'}}" class="img-fluid rounded border-2 border border-primary">
                        </span>
                        <span class="hide-mob ms-2">{{__('Hi, ')}}{{\Auth::user()->name }}!</span>
                        <i class="ti ti-chevron-down drp-arrow nocolor hide-mob"></i>
                    </a>
                    <div class="dropdown-menu dash-h-dropdown">

                        <a href="{{route('profile')}}" class="dropdown-item">
                            <i class="ti ti-user text-dark"></i><span>{{__('Profile')}}</span>
                        </a>

                        <a href="{{ route('logout') }}" onclick="event.preventDefault(); document.getElementById('frm-logout').submit();" class="dropdown-item">
                            <i class="ti ti-power text-dark"></i><span>{{__('Logout')}}</span>
                        </a>

                        <form id="frm-logout" action="{{ route('logout') }}" method="POST" class="d-none">
                            {{ csrf_field() }}
                        </form>

                    </div>
                </li>

            </ul>
        </div>
        <div class="ms-auto">
            <ul class="list-unstyled">
                @if(\Auth::user()->type == 'company' )
                @impersonating($guard = null)
                <li class="dropdown dash-h-item drp-company">
                    <a class="btn btn-danger btn-sm me-3" href="{{ route('exit.company') }}"><i class="ti ti-ban"></i>
                        {{ __('Exit Company Login') }}
                    </a>
                </li>
                @endImpersonating
                @endif

                @if( \Auth::user()->type !='client' && \Auth::user()->type !='super admin' )
                    <li class="dropdown dash-h-item drp-notification">
                        <a class="dash-head-link arrow-none me-0" href="{{ url('chats') }}" aria-haspopup="false"
                           aria-expanded="false">
                            <i class="ti ti-brand-hipchat"></i>
                            <span class="bg-danger dash-h-badge message-toggle-msg  message-counter custom_messanger_counter beep"> {{ $unseenCounter }}<span
                                    class="sr-only"></span>
                            </span>
                        </a>
                    </li>
                @endif

                {{-- AI Agent Token Display --}}
                @php
                    // This logic block is now slightly redundant as it's also in admin.blade.php
                    // Consider refactoring into a View Composer or Service Provider later if needed
                    $ai_tokens_remaining_header = 0; // Use a different variable name to avoid conflict
                    $company_user_header = null;
                    $plan_header = null;
                    $ai_feature_enabled_header = false;
                    
                    $users_header = \Auth::user(); // Need to get user again here
                    if ($users_header) {
                        if ($users_header->type == 'company') {
                            $company_user_header = $users_header;
                        } elseif ($users_header->created_by) {
                            $company_user_header = App\Models\User::find($users_header->created_by);
                        }
                
                        if ($company_user_header && $company_user_header->type == 'company') {
                            // Get the current plan
                            $plan_header = \App\Models\Plan::find($company_user_header->plan);
                            
                            // Check if the plan has AI agent enabled
                            $ai_feature_enabled_header = $plan_header && $plan_header->ai_agent_enabled == 1;
                            
                            // Calculate tokens remaining based on plan allocation and user usage
                            $tokens_allocated_header = $plan_header ? $plan_header->ai_agent_default_tokens : 0;
                            $ai_tokens_remaining_header = max(0, $tokens_allocated_header - $company_user_header->ai_agent_tokens_used);
                        }
                    }
                @endphp
                
                @if($ai_feature_enabled_header)
                    <li class="dash-h-item" id="header-token-display">
                        <a href="{{ route('ai_agent.chat.show') }}" class="dash-head-link" title="{{ __('AI Agent Tokens') }}">
                            <i class="ti ti-robot text-primary"></i>
                            <span class="badge bg-primary text-dark dark:text-white ms-1 px-2 py-1 rounded-pill {{ $ai_tokens_remaining_header > 0 ? '' : 'opacity-75' }} remaining-tokens">{{ number_format($ai_tokens_remaining_header) }}</span>
                        </a>
                    </li>
                @endif
                {{-- End AI Agent Token Display --}}

                <li class="dropdown dash-h-item drp-language">
                    <a
                        class="dash-head-link dropdown-toggle arrow-none me-0"
                        data-bs-toggle="dropdown"
                        href="#"
                        role="button"
                        aria-haspopup="false"
                        aria-expanded="false"
                    >
                        <i class="ti ti-world nocolor"></i>
                        <span class="drp-text hide-mob">{{ucfirst($LangName->full_name)}}</span>
                        <i class="ti ti-chevron-down drp-arrow nocolor"></i>
                    </a>
                    <div class="dropdown-menu dash-h-dropdown dropdown-menu-end">
                        @foreach ($languages as $code => $language)
                            <a href="{{ route('change.language', $code) }}"
                               class="dropdown-item {{ $lang == $code ? 'text-primary' : '' }}">
                                <span>{{ucFirst($language)}}</span>
                            </a>
                        @endforeach

                        <h></h>
                            @if(\Auth::user()->type=='super admin')
                                <a data-url="{{ route('create.language') }}" class="dropdown-item text-primary" data-ajax-popup="true" data-title="{{__('Create New Language')}}" style="cursor: pointer">
                                    {{ __('Create Language') }}
                                </a>
                                <a class="dropdown-item text-primary" href="{{route('manage.language',[isset($lang)?$lang:'english'])}}">{{ __('Manage Language') }}</a>
                            @endif
                    </div>
                </li>
            </ul>
        </div>
    </div>
    </header>
