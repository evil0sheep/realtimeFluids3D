//callbacksPBO.cpp (Rob Farber)

#include <GL/glut.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <stdlib.h>
//#include "raytrace.hpp"
#include "glm/glm.hpp"

// variables for keyboard control
int animFlag=1;
float animTime=0.0f;
float animInc=0.1f;

//external variables
extern GLuint pbo;
extern GLuint textureID;
extern unsigned int image_width;
extern unsigned int image_height;
extern void moveIn();
extern void moveOut();
extern void moveUp();
extern void moveDown();
extern void moveLeft();
extern void moveRight();

glm::mat4 camTrans(1);


// The user must create the following routines:
void runCuda();

void display()
{
   // run CUDA kernel
   runCuda();

   // Create a texture from the buffer
   glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);

   // bind texture from PBO
   glBindTexture(GL_TEXTURE_2D, textureID);


   // Note: glTexSubImage2D will perform a format conversion if the
   // buffer is a different format from the texture. We created the
   // texture with format GL_RGBA8. In glTexSubImage2D we specified
   // GL_BGRA and GL_UNSIGNED_INT. This is a fast-path combination

   // Note: NULL indicates the data resides in device memory
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height,
         GL_RGBA, GL_UNSIGNED_BYTE, NULL);


   // Draw a single Quad with texture coordinates for each vertex.

   glBegin(GL_QUADS);
   glTexCoord2f(0.0f,1.0f); glVertex3f(0.0f,0.0f,0.0f);
   glTexCoord2f(0.0f,0.0f); glVertex3f(0.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,0.0f); glVertex3f(1.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,1.0f); glVertex3f(1.0f,0.0f,0.0f);
   glEnd();

   // Don't forget to swap the buffers!
   glutSwapBuffers();

   // if animFlag is true, then indicate the display needs to be redrawn
   if(animFlag) {
      glutPostRedisplay();
      animTime += animInc;
   }
}



// No mouse event handlers defined
void mouse(int button, int state, int x, int y)
{

	
}

void motion(int x, int y)
{
}

/*
void mouse(int button, int state, int x, int y) {
  unsigned char color[3];

  if (button == GLUT_LEFT_BUTTON) {
	 if (state == GLUT_DOWN) { 
	  //printf("mouse clicked at %d %d\n", x, y);

	}
  }
}


vec2 transform_point(vec2 in) {
	float c, d, e, f;
	if (g_width > g_height) {
			d = (g_width - 1) / 2.0;
			c = (g_height * (g_width - 1.0))/ (2.0 * g_width);
			f = (g_height - 1.0) / 2.0;
			e = (g_height - 1.0) / 2.0;
	} else {
		d = (g_width - 1.0) / 2.0;
		c = (g_width - 1.0) / 2.0;
		f = (g_height - 1.0) / 2.0;
		e = (g_width  * (g_height - 1.0))/ (2.0 * g_height);
		}		
		
		
	//return vec2(c * in.x + d, e * in.x + f);
	return vec2( (in.x - d) / c, -((in.y - f) / e));
} 

//the mouse move callback
void motion(int x, int y) {
	vec2 new_world=transform_point(vec2(x, y));

	int d_x=x-g_x;
	int d_y=-(y-g_y);


  	if(abs(g_width/2-x)>3 || abs(g_height/2-y)>3) {
  

 		g_player.transform = glm::rotate(g_player.transform, (g_width/2-x)*g_mouse_speed,vec3(0,1,0));
 		g_cam.transform = glm::rotate(mat4(1), (g_height/2-y)*g_mouse_speed, vec4to3(g_cam.transform*vec4(1,0,0,0)))*g_cam.transform ;
  	
		glutPostRedisplay();
	
		glutWarpPointer(g_width/2, g_height/2);
	}
}*/
