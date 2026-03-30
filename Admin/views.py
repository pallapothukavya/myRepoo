from django.shortcuts import render

# Create your views here.
# Create your views here.
from django.shortcuts import render,redirect
from django.contrib import messages
from users.models import UserRegistrationModel


# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return redirect('AdminHome')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})



def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewregisterusers.html', context={'data': data})




def ActivaUsers(request):
    if request.method == 'GET':
        user_id = request.GET.get('uid')
        
        if user_id:  # Ensure user_id is not None
            status = 'activated'
            print("Activating user with ID =", user_id)
            UserRegistrationModel.objects.filter(id=user_id).update(status=status)

        # Redirect to the view where users are listed after activation
        return redirect('RegisterUsersView')  # Replace with your actual URL name

def DeleteUsers(request):
    if request.method == 'GET':
        user_id = request.GET.get('uid')
        
        if user_id:  # Ensure user_id is not None
            print("Deleting user with ID =", user_id)
            UserRegistrationModel.objects.filter(id=user_id).delete()

        # Redirect to the view where users are listed after deletion
        return redirect('RegisterUsersView')  # Replace with your actual URL name


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def api_admin_login(request):
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
            usrid = data.get('loginid')
            pswd = data.get('password')
        except:
            usrid = request.POST.get('loginid')
            pswd = request.POST.get('password')

        if usrid == 'admin' and pswd == 'admin':
            return JsonResponse({'status': 'success', 'message': 'Admin login successful'})
        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid admin credentials'}, status=401)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)
